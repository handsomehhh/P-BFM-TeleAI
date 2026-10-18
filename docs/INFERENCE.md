# Inference and deployment export

The policy inference entry points are:

```bash
python -m pbfm.inference.tracking --help
python -m pbfm.inference.goal --help
python -m pbfm.inference.reward --help
python -m pbfm.inference.play --help      # interactive browser viewer
```

Run from the repository root after installation. Supply the run directory with
`--model-folder`; it must contain the selected checkpoint. Motion data and weights
are external. Tracking includes camera, terrain selection, terrain-relative reset
and motion-reference/policy comparison options. For a training terrain tile, for
example:

```bash
python -m pbfm.inference.tracking --model-folder /path/to/run \
  --motion-list 20 --save-mp4 \
  --terrain-family low_stairs_down --difficulty-row 7
```

Terrain geometry is read from that checkpoint's configuration. The family name
alone does not specify the direction of travel: RP1 center-outward traversal of
`low_stairs_down` goes uphill. Check reset position, heading and actual geometry
before labelling a video.

## Interactive playback

`pbfm.inference.play` runs the same tracking rollout and streams it to a browser
(Viser), with the reference motion drawn as a translucent ghost. Motion, RP1
terrain family/difficulty row, speed and frame are chosen on the page, and
"Reload latest checkpoint" picks up a newer checkpoint from a run that is still
training. The checkpoint is copied to a local snapshot before loading, so a run
that is rewriting `checkpoint/` is safe to point at.

```bash
python -m pbfm.inference.play --model-folder /path/to/run \
  --motion-id 20 --terrain-family flat --port 16180
# on your workstation:
ssh -N -L 16180:127.0.0.1:16180 <cluster>   # then open http://localhost:16180
```

Each (motion, terrain) episode is simulated once, shown live while it runs, then
replayed in real time with a scrubbable timeline; finished episodes are cached
until the checkpoint is reloaded. Playback is deterministic (mean actions, domain
randomization and observation noise off by default). The single-environment
observation pipeline is slower than real time, especially on a GPU shared with
training (about 8 steps/s next to a full training job on an H100), which is why
playback is recorded first rather than driven live.

## Context

Tracking advances an eight-frame reference latent window (shortened at the end).
Heading and valid planar-velocity targets use the same aligned reference. A valid
zero-velocity target means stop; an invalid target means no velocity instruction.
Goal and reward latents do not automatically create reference velocity targets.
Use the CLI's velocity-context options deliberately for custom motion names.

Keep the exported latent, heading/velocity fields, validity flags, frame rate and
coordinate convention together. A latent alone is not the complete input contract
of a command-conditioned policy. The policy always uses its estimated current
velocity, not simulator velocity labels, during deployment.

## Export

```bash
python -m scripts.export_depth_policy_onnx \
  --checkpoint /path/to/run/checkpoint --output /path/to/export/policy.onnx
python -m scripts.export_backward_encoder_onnx \
  --model-folder /path/to/run --output /path/to/export/backward_encoder.onnx --device cpu
python -m scripts.verify_depth_policy_trace --help
```

The exported `policy.onnx` includes **depth encoding, the velocity
estimator, observation normalization and the Actor**. It is not a split actor-only
graph. Backward encoding is a separate export. No critic or simulator-only velocity
label is an input to the deployed policy. Inspect the generated metadata for the
precise names and shapes instead of reusing another checkpoint's interface.

ONNX receives normalized float depth in [0,1]. Environment uint8 depth must be
converted to float32 and divided by 255 once; already-normalized float depth must
not be divided again. Real camera preprocessing, temporal history, context frame
alignment, action scaling and PD control remain outside the graph. ONNX parity
tests do not validate those real-robot integration steps.
