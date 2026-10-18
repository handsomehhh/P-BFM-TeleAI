"""Interactive browser playback of a PBFM tracking policy (Viser).

Runs the same MJLab/MuJoCo-Warp rollout as ``pbfm.inference.tracking`` and
streams it to a browser, with the reference motion drawn as a translucent ghost.
Motion, RP1 terrain tile, playback speed and checkpoint are chosen on the page.

The rollout is deterministic (mean actions, no domain randomization or
observation noise by default), so each (motion, terrain) episode is simulated
once, shown live while it runs, recorded, and then replayed in real time with a
scrubbable timeline. The single-env observation pipeline is far slower than
real time on a GPU that is shared with training; recording keeps playback
smooth regardless. Finished rollouts are cached per checkpoint.

The checkpoint is copied to a local snapshot before loading, so a training run
that is rewriting ``checkpoint/`` cannot hand the viewer a half-written model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import viser
from torch.utils._pytree import tree_map

from mjlab.viewer.viser.scene import MjlabViserScene

from pbfm.agents.behavior_context import heading_observation, root_heading_xy
from pbfm.agents.inference_context import build_tracking_commands
from pbfm.agents.load_utils import load_model_from_checkpoint_dir
from pbfm.inference.runtime import (
    _assign_rp1_training_tile,
    _control_to_qpos_order_indices,
    _expert_qpos_from_obs,
    _target_states_from_obs,
    _terrain_env_cfg,
    _tracking_z,
    add_bool_arg,
    add_robot_config_manifest_args,
    checkpoint_load_device,
    load_mjlab_env_cfg,
    resolve_inference_data_and_robot_args,
    resolve_inference_robot_config,
)
from pbfm.inference.tracking import _center_target_states_on_terrain
from pbfm.robots import load_robot_training_spec
from pbfm.terrains.rp1_simple import RP1_TERRAIN_COMPONENT_NAMES
from pbfm.utils.helpers import get_backward_observation

RP1_TERRAIN_SEED = 4831
LIVE_RENDER_FPS = 10.0


def _file_signature(folder: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(folder)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def snapshot_checkpoint(model_folder: Path, dest: Path, *, retries: int = 10, wait_s: float = 20.0) -> dict[str, Any]:
    """Copy ``checkpoint/model`` to ``dest`` once no file changes during the copy.

    Returns the checkpoint's train status (empty if the folder has none).
    """
    src = model_folder / "checkpoint"
    status_path = src / "train_status.json"
    for attempt in range(retries):
        before = (_file_signature(src / "model"), status_path.read_text() if status_path.exists() else "")
        target = dest / "checkpoint" / "model"
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(src / "model", target)
        after = (_file_signature(src / "model"), status_path.read_text() if status_path.exists() else "")
        if before == after:
            return json.loads(after[1]) if after[1] else {}
        print(f"[INFO] Checkpoint changed while copying (attempt {attempt + 1}); retrying in {wait_s:.0f}s", flush=True)
        time.sleep(wait_s)
    raise RuntimeError(f"Checkpoint under {src} kept changing; could not take a consistent snapshot")


@dataclass
class Episode:
    """One rollout: live policy state while simulating, then its recorded frames."""

    key: tuple
    motion_id: int
    z: torch.Tensor
    commands: Any
    heading_targets: torch.Tensor | None
    ghost_qpos: np.ndarray
    frames: list[dict[str, np.ndarray]] = field(default_factory=list)
    step: int = 0
    ended: str | None = None

    @property
    def total_steps(self) -> int:
        return int(self.z.shape[0])


class PlaySession:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = args.device
        self.model_folder = args.model_folder.expanduser().resolve()
        self.snapshot_root = Path(tempfile.mkdtemp(prefix="pbfm_play_"))

        robot_config = resolve_inference_robot_config(args.robot_config, None)
        self.robot_training = load_robot_training_spec(robot_config)
        self.num_dof = len(self.robot_training.robot.control_joint_names)
        self.dof_qpos_order, self.qpos_joint_names = _control_to_qpos_order_indices(self.robot_training)

        self.model = None
        self.checkpoint_label = ""
        self.cache: dict[tuple, Episode] = {}
        self.load_checkpoint()

        env_cfg, self.use_root_height_obs = load_mjlab_env_cfg(
            self.model_folder,
            data_path=args.data_path,
            robot_config=robot_config,
            device=self.device,
            headless=True,
            disable_dr=args.disable_dr,
            disable_obs_noise=args.disable_obs_noise,
            max_episode_length_s=args.max_episode_length_s,
        )
        # Every RP1 family lives in the same seeded grid, so a tile change is only
        # an env-origin reassignment, never an environment rebuild.
        self.env_cfg = _terrain_env_cfg(env_cfg, args.terrain_family, seed=RP1_TERRAIN_SEED)
        self.resources = ExitStack()
        self.wrapped_env = self.env_cfg.build(num_envs=1)[0]
        self.resources.callback(self.wrapped_env.close)
        self.env = self.wrapped_env._env
        self.env._motion_lib.load_all_motions()
        self.env.is_evaluating = True
        self.dt = float(self.env.dt)
        self.motion_names = list(self.env._motion_lib._motion_data_keys)
        self.num_rows = int(self.env.mjlab_env.scene["terrain"].terrain_origins.shape[0])
        self.sim = self.env.mjlab_env.sim
        self.robot_qpos_slice = self._robot_qpos_slice()

        self.lock = threading.Lock()
        self.pending_reset = True
        self.paused = False
        self.speed = 1.0
        self.loop = True
        self.show_ghost = True
        self.motion_id = int(args.motion_id)
        self.terrain_family = args.terrain_family
        self.difficulty_row = int(args.difficulty_row)
        self.episode: Episode | None = None
        self.observation: dict[str, torch.Tensor] | None = None
        self.cursor = 0.0  # replay frame position
        self.seek_to: int | None = None
        self.sim_fps = 0.0

    # Checkpoint.

    def load_checkpoint(self) -> None:
        status = snapshot_checkpoint(self.model_folder, self.snapshot_root)
        model = load_model_from_checkpoint_dir(self.snapshot_root / "checkpoint", device=checkpoint_load_device(self.device))
        model.to(self.device)
        model.eval()
        self.model = model
        self.cache.clear()
        step = status.get("global_time")
        self.checkpoint_label = f"{step / 1e6:.1f}M steps" if step is not None else self.model_folder.name
        print(f"[INFO] Loaded checkpoint {self.model_folder} ({self.checkpoint_label})", flush=True)

    # Scene geometry.

    def _robot_qpos_slice(self) -> slice:
        """Locate the robot's [root(7) + joints] block inside the scene qpos."""
        m = self.sim.mj_model
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise RuntimeError(f"Expected one free joint in the play scene, found {len(free)}")
        start = int(m.jnt_qposadr[free[0]])
        names = [m.joint(free[0] + 1 + i).name for i in range(self.num_dof)]
        if [n.split("/")[-1] for n in names] != list(self.qpos_joint_names):
            raise RuntimeError(f"Scene joint order {names} does not match robot qpos order {self.qpos_joint_names}")
        return slice(start, start + 7 + self.num_dof)

    def record_frame(self) -> dict[str, np.ndarray]:
        d, m = self.sim.data, self.sim.mj_model
        frame = {
            "xpos": d.xpos.cpu().numpy(),
            "xmat": d.xmat.cpu().numpy(),
            "qpos": d.qpos.cpu().numpy(),
            "qvel": d.qvel.cpu().numpy(),
        }
        if m.nu > 0:
            frame["ctrl"] = d.ctrl.cpu().numpy()
        if m.nmocap > 0:
            frame["mocap_pos"] = d.mocap_pos.cpu().numpy()
            frame["mocap_quat"] = d.mocap_quat.cpu().numpy()
        return frame

    # Episodes.

    def episode_key(self) -> tuple:
        return (self.motion_id, self.terrain_family, self.difficulty_row)

    def start_episode(self) -> None:
        key = self.episode_key()
        self.cursor = 0.0
        if key in self.cache:
            self.episode = self.cache[key]
            return
        env, device = self.env, self.device
        _assign_rp1_training_tile(env, family=self.terrain_family, difficulty_row=self.difficulty_row)
        motion_id = self.motion_id
        with torch.no_grad():
            backward_obs, obs_dict = get_backward_observation(env, motion_id, use_root_height_obs=self.use_root_height_obs)
            z = _tracking_z(self.model, tree_map(lambda x: x[1:].to(device) if hasattr(x, "to") else x, backward_obs))
        target_states = _target_states_from_obs(obs_dict, device=device, num_dof=self.num_dof)
        target_states = _center_target_states_on_terrain(target_states, env, terrain_relative_height=True)
        self.observation, _ = self.wrapped_env.reset(to_numpy=False, target_states=target_states)
        commands = build_tracking_commands(
            root_heading_xy(obs_dict["ref_body_rots"][:, 0].to(device).float()),
            obs_dict["ref_body_vels"][:, 0, :2].to(device).float(),
            root_heading_xy(env.base_quat.float()),
            motion_name=self.motion_names[motion_id],
            frames=len(z),
            mode=self.args.velocity_context,
            policy=getattr(self.model.cfg, "reference_velocity_policy", "walk_run"),
            temporal_mode=getattr(self.model.cfg, "reference_velocity_temporal_mode", "instantaneous"),
            temporal_horizon=int(getattr(self.model.cfg, "reference_velocity_temporal_horizon", 1)),
        )
        heading_targets = commands.heading_world if getattr(self.model.cfg, "heading_context_enabled", False) else None
        ghost_qpos = _expert_qpos_from_obs(obs_dict, num_dof=self.num_dof, dof_qpos_order_indices=self.dof_qpos_order)
        # Move the reference to where the rollout was reset (tile centre / height).
        ghost_qpos[:, :3] += target_states["root_states"][0, :3].cpu().numpy() - ghost_qpos[0, :3]
        self.episode = Episode(key, motion_id, z, commands, heading_targets, ghost_qpos)
        self.episode.frames.append(self.record_frame())

    def step_policy(self) -> None:
        ep, env, device = self.episode, self.env, self.device
        obs = self.observation
        current_heading = root_heading_xy(env.base_quat.float())
        if ep.heading_targets is not None:
            target = ep.heading_targets[min(ep.step, ep.heading_targets.shape[0] - 1)].unsqueeze(0)
            valid = torch.ones((1, 1), device=device, dtype=torch.bool)
            obs["heading"] = heading_observation(current_heading, target, valid)
        if getattr(self.model.cfg, "velocity_critic_enabled", False):
            obs["velocity_command"] = ep.commands.velocity_at(ep.step, current_heading)
        with torch.no_grad():
            action = self.model.act(obs, ep.z[ep.step].unsqueeze(0), mean=True)
        self.observation, _r, terminated, truncated, _info = self.wrapped_env.step(action, to_numpy=False)
        ep.step += 1
        ep.frames.append(self.record_frame())
        if bool(torch.as_tensor(terminated).any()):
            ep.ended = "terminated (fell or lost track)"
        elif bool(torch.as_tensor(truncated).any()):
            ep.ended = "truncated"
        elif ep.step >= ep.total_steps:
            ep.ended = "motion finished"
        if ep.ended is not None:
            self.cache[ep.key] = ep

    @property
    def simulating(self) -> bool:
        return self.episode is not None and self.episode.ended is None

    # Rendering.

    def render(self, scene: MjlabViserScene, frame_idx: int) -> None:
        ep = self.episode
        frame = ep.frames[frame_idx]
        scene.clear()
        if self.show_ghost:
            qpos = frame["qpos"][0].copy()
            qpos[self.robot_qpos_slice] = ep.ghost_qpos[min(frame_idx, len(ep.ghost_qpos) - 1)]
            scene.add_ghost_mesh(qpos, self.sim.mj_model, alpha=0.35, label="reference")
        scene.update_from_arrays(
            frame["xpos"],
            frame["xmat"],
            frame.get("mocap_pos"),
            frame.get("mocap_quat"),
            0,
            qpos=frame["qpos"],
            qvel=frame["qvel"],
            ctrl=frame.get("ctrl"),
        )

    def status_html(self) -> str:
        ep = self.episode
        if ep is None:
            return "Loading..."
        if self.simulating:
            state = f"simulating {ep.step}/{ep.total_steps} at {self.sim_fps:.0f} steps/s (live view; replay starts when done)"
        else:
            frame = int(self.cursor)
            state = (f"replay {frame}/{len(ep.frames) - 1} ({frame * self.dt:.1f}s), "
                     f"{'paused' if self.paused else f'{self.speed:g}x'}")
        lines = [
            f"<b>Checkpoint</b>: {self.checkpoint_label}",
            f"<b>Motion</b>: {ep.motion_id} {self.motion_names[ep.motion_id]} ({ep.total_steps * self.dt:.1f}s)",
            f"<b>Terrain</b>: {self.terrain_family}, row {self.difficulty_row}",
            f"<b>State</b>: {state}",
        ]
        if ep.ended:
            lines.append(f"<b>Episode</b>: {ep.ended} after {ep.step * self.dt:.1f}s")
        return '<div style="font-size:0.85em;line-height:1.4;padding:0 1em">' + "<br/>".join(lines) + "</div>"

    # GUI.

    def build_gui(self, server: viser.ViserServer, scene: MjlabViserScene) -> None:
        gui = server.gui
        tabs = gui.add_tab_group()
        with tabs.add_tab("Play", icon=viser.Icon.PLAYER_PLAY):
            self.status = gui.add_html(self.status_html())
            with gui.add_folder("Playback"):
                self.timeline = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
                pause = gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
                frame_step = gui.add_button_group("", options=["< Frame", "Frame >"])
                restart = gui.add_button("Re-simulate", icon=viser.Icon.REFRESH,
                                         hint="Drop the cached rollout and simulate this motion again.")
                speed = gui.add_slider("Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0)
                loop = gui.add_checkbox("Loop", initial_value=True)
                ghost = gui.add_checkbox("Reference ghost", initial_value=True)
            with gui.add_folder("Motion"):
                labels = [f"{i}: {n}" for i, n in enumerate(self.motion_names)]
                motion = gui.add_dropdown("Motion", options=labels, initial_value=labels[self.motion_id])
                prev_next = gui.add_button_group("", options=["Prev", "Next"])
            with gui.add_folder("Terrain"):
                family = gui.add_dropdown("Family", options=list(RP1_TERRAIN_COMPONENT_NAMES), initial_value=self.terrain_family)
                row = gui.add_slider("Difficulty row", min=0, max=self.num_rows - 1, step=1, initial_value=self.difficulty_row)
            with gui.add_folder("Checkpoint"):
                reload_btn = gui.add_button("Reload latest checkpoint", icon=viser.Icon.DATABASE)
            with gui.add_folder("Scene"):
                scene.create_scene_gui(
                    camera_distance=self.args.camera_distance,
                    camera_azimuth=self.args.camera_azimuth,
                    # --camera-elevation uses MuJoCo's sign (negative looks down); mjviser's is the opposite.
                    camera_elevation=-self.args.camera_elevation,
                    show_debug_viz_control=False,
                )
        with tabs.add_tab("Visualization", icon=viser.Icon.EYE):
            scene.create_overlay_gui()
        with tabs.add_tab("Groups", icon=viser.Icon.LAYERS_INTERSECT):
            scene.create_groups_gui()

        self._timeline_programmatic = False

        @self.timeline.on_update
        def _(_) -> None:
            if not self._timeline_programmatic:
                self.seek_to = int(self.timeline.value)

        @pause.on_click
        def _(_) -> None:
            self.paused = not self.paused
            pause.label = "Play" if self.paused else "Pause"
            pause.icon = viser.Icon.PLAYER_PLAY if self.paused else viser.Icon.PLAYER_PAUSE

        @frame_step.on_click
        def _(event) -> None:
            self.paused = True
            pause.label, pause.icon = "Play", viser.Icon.PLAYER_PLAY
            self.seek_to = int(self.cursor) + (1 if event.target.value == "Frame >" else -1)

        @restart.on_click
        def _(_) -> None:
            with self.lock:
                self.cache.pop(self.episode_key(), None)
                self.pending_reset = True

        @speed.on_update
        def _(_) -> None:
            self.speed = float(speed.value)

        @loop.on_update
        def _(_) -> None:
            self.loop = bool(loop.value)

        @ghost.on_update
        def _(_) -> None:
            self.show_ghost = bool(ghost.value)
            self.seek_to = int(self.cursor)  # re-render the current frame

        @motion.on_update
        def _(_) -> None:
            with self.lock:
                self.motion_id = int(motion.value.split(":", 1)[0])
                self.pending_reset = True

        @prev_next.on_click
        def _(event) -> None:
            new_id = (self.motion_id + (1 if event.target.value == "Next" else -1)) % len(self.motion_names)
            motion.value = labels[new_id]  # selection is applied by motion.on_update

        @family.on_update
        def _(_) -> None:
            with self.lock:
                self.terrain_family = family.value
                self.pending_reset = True

        @row.on_update
        def _(_) -> None:
            with self.lock:
                self.difficulty_row = int(row.value)
                self.pending_reset = True

        @reload_btn.on_click
        def _(_) -> None:
            reload_btn.disabled = True
            try:
                with self.lock:
                    self.load_checkpoint()
                    self.pending_reset = True
            except Exception as exc:  # keep the current model on a failed reload
                print(f"[WARN] Checkpoint reload failed: {exc}", flush=True)
            finally:
                reload_btn.disabled = False

    def set_timeline(self, value: int, maximum: int) -> None:
        self._timeline_programmatic = True
        try:
            if self.timeline.max != maximum:
                self.timeline.max = max(1, maximum)
            if self.timeline.value != value:
                self.timeline.value = value
        finally:
            self._timeline_programmatic = False

    # Main loop.

    def run(self) -> None:
        server = viser.ViserServer(host=self.args.host, port=self.args.port, label="PBFM play")
        scene = MjlabViserScene(
            server=server,
            mj_model=self.sim.mj_model,
            num_envs=1,
            sim_model=self.sim.model,
            expanded_fields=self.sim.expanded_fields,
        )
        scene.debug_visualization_enabled = True  # required for the reference ghost
        self.build_gui(server, scene)
        print(f"[INFO] PBFM play viewer at http://{self.args.host}:{self.args.port}", flush=True)

        last_render = last_status = last_log = fps_t0 = last_tick = time.perf_counter()
        fps_steps = 0
        try:
            while True:
                now = time.perf_counter()
                elapsed, last_tick = now - last_tick, now
                with self.lock:
                    if self.pending_reset:
                        self.pending_reset = False
                        self.seek_to = None
                        self.start_episode()
                        self.render(scene, 0)
                    ep = self.episode
                    if self.simulating:
                        self.step_policy()
                        fps_steps += 1
                        if now - last_render >= 1.0 / LIVE_RENDER_FPS or not self.simulating:
                            self.cursor = float(len(ep.frames) - 1) if self.simulating else 0.0
                            self.render(scene, int(self.cursor))
                            last_render = now
                    else:
                        last_frame = len(ep.frames) - 1
                        prev = int(self.cursor)
                        if self.seek_to is not None:
                            self.cursor = float(min(max(self.seek_to, 0), last_frame))
                            self.seek_to = None
                        elif not self.paused:
                            self.cursor += elapsed * self.speed / self.dt
                            if self.cursor > last_frame:
                                self.cursor = 0.0 if self.loop else float(last_frame)
                        if int(self.cursor) != prev or now - last_render >= 1.0:
                            self.render(scene, int(self.cursor))
                            last_render = now
                if now - fps_t0 >= 1.0:
                    self.sim_fps = fps_steps / (now - fps_t0)
                    fps_t0, fps_steps = now, 0
                if now - last_status >= 0.25:
                    self.status.content = self.status_html()
                    self.set_timeline(int(self.cursor), len(ep.frames) - 1)
                    last_status = now
                if now - last_log >= 10.0:
                    phase = f"simulating {ep.step}/{ep.total_steps} sim={self.sim_fps:.0f} steps/s" if self.simulating \
                        else f"replay {int(self.cursor)}/{len(ep.frames) - 1}"
                    print(f"[PLAY] motion={ep.motion_id} terrain={self.terrain_family}:{self.difficulty_row} "
                          f"{phase} ended={ep.ended} cached={len(self.cache)}", flush=True)
                    last_log = now
                if not self.simulating:
                    time.sleep(1.0 / self.args.render_fps)
        except KeyboardInterrupt:
            pass
        finally:
            server.stop()
            self.close()

    def close(self) -> None:
        self.resources.close()
        shutil.rmtree(self.snapshot_root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive browser playback of a PBFM tracking policy.")
    parser.add_argument("--model-folder", type=Path, required=True, help="Run directory containing config.json and checkpoint/.")
    parser.add_argument("--data-path", type=Path, default=None)
    add_robot_config_manifest_args(parser, purpose="interactive playback")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; keep 127.0.0.1 and use an SSH tunnel.")
    parser.add_argument("--port", type=int, default=16180)
    parser.add_argument("--motion-id", type=int, default=20)
    parser.add_argument("--terrain-family", choices=RP1_TERRAIN_COMPONENT_NAMES, default="flat")
    parser.add_argument("--difficulty-row", type=int, default=7)
    add_bool_arg(parser, "--disable-dr", True, "Disable domain randomization (default on for clean, repeatable playback).")
    add_bool_arg(parser, "--disable-obs-noise", True, "Disable observation noise (default on for clean, repeatable playback).")
    parser.add_argument("--velocity-context", choices=("auto", "valid", "invalid"), default="auto")
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    parser.add_argument("--render-fps", type=float, default=50.0, help="Replay frame rate pushed to the browser.")
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    return resolve_inference_data_and_robot_args(parser.parse_args(), parser)


def main() -> None:
    args = parse_args()
    session = PlaySession(args)
    if not 0 <= args.motion_id < len(session.motion_names):
        session.close()
        raise SystemExit(f"--motion-id must be in [0, {len(session.motion_names)})")
    if not 0 <= args.difficulty_row < session.num_rows:
        session.close()
        raise SystemExit(f"--difficulty-row must be in [0, {session.num_rows})")
    session.run()


if __name__ == "__main__":
    main()
