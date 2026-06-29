# AgileX Cobot Magic Client

This client runs on the **AgileX Cobot Magic upper computer** and connects to the NIAF WebSocket server through the OpenPI client protocol.

It supports:

- AgileX Cobot Magic real-robot I/O.
- Position control and MIT impedance control.
- Synchronous action chunk execution.
- Asynchronous inference with chunk prefetch and blend-window smoothing.
- Optional runtime monitoring for MIT-control experiments.

## Environment

Use a uv-managed robot/client environment on the Cobot Magic upper computer. This follows OpenPI's real-robot examples: create a Python 3.10 venv, sync the robot-client requirements, then install `openpi-client` into that venv. The policy server uses the separate conda environment documented at the repository root.

If the whole `NIAF.real` repository is available on the upper computer:

```bash
cd /path/to/NIAF.real
uv venv --python 3.10 client/agilex_cobot_magic/.venv
source client/agilex_cobot_magic/.venv/bin/activate
uv pip sync client/agilex_cobot_magic/requirements.txt
uv pip install -e third_party/openpi/packages/openpi-client
export PYTHONPATH=$PWD/client:$PWD/third_party/openpi/packages/openpi-client/src:$PYTHONPATH
```

If this directory is copied standalone to the upper computer with a local `packages/openpi-client` directory, install that package instead:

```bash
cd /path/to/agilex_cobot_magic
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip sync requirements.txt
uv pip install -e packages/openpi-client
export PYTHONPATH=$PWD:$PWD/packages/openpi-client/src:$PYTHONPATH
```

The upper computer must also provide ROS, camera drivers, `cv_bridge`, socket-CAN setup, and the AgileX Piper SDK (`piper_sdk`) used by `mit_controller.py` and MIT-mode monitoring. `requirements.txt` covers the pip-installable OpenPI real-robot client stack; ROS workspace setup and vendor SDK installation are still external prerequisites.

## Run

The client has two independent mode switches:

- Control mode: `--control-mode position` or `--control-mode mit`.
- Inference mode: synchronous by default, asynchronous with `--async-infer`.

Use `position` for the ROS topic position-control path. Use `mit` for direct MIT impedance control; this mode can consume server-provided joint velocities when the server returns them.

Use synchronous inference first when testing a new checkpoint or a new robot setup. Use asynchronous inference for real runs where chunk-to-chunk pauses are visible; it requests the next chunk in the background and blends into the new chunk.

### 1. Position Control + Synchronous Inference

Safest basic smoke test. The robot executes one action chunk, then blocks while waiting for the next server response.

```bash
python -m agilex_cobot_magic.main \
  --host <server-ip> \
  --port 8000 \
  --control-mode position
```

### 2. Position Control + Asynchronous Inference

Keeps the ROS position-control path, but overlaps execution and inference.

```bash
python -m agilex_cobot_magic.main \
  --host <server-ip> \
  --port 8000 \
  --control-mode position \
  --async-infer \
  --trigger-step 15 \
  --blend-window 10
```

### 3. MIT Impedance Control + Synchronous Inference

Uses direct MIT impedance control, but still waits between chunks. This is useful for validating MIT parameters before enabling async execution.

```bash
python -m agilex_cobot_magic.main \
  --host <server-ip> \
  --port 8000 \
  --control-mode mit \
  --can-port-left can_left \
  --can-port-right can_right \
  --mit-kp 10.0 \
  --mit-kd 0.8 \
  --record
```

### 4. MIT Impedance Control + Asynchronous Inference

Recommended real-run mode after position control, MIT control, and server velocity output have been validated.

```bash
python -m agilex_cobot_magic.main \
  --host <server-ip> \
  --port 8000 \
  --control-mode mit \
  --can-port-left can_left \
  --can-port-right can_right \
  --mit-kp 10.0 \
  --mit-kd 0.8 \
  --async-infer \
  --trigger-step 15 \
  --blend-window 10 \
  --record
```

The same code can also be run from this directory with `python main.py` if the uv environment is activated and the OpenPI client package is importable.

## Mode Options

- `--control-mode position`: publish position targets through the normal robot command path.
- `--control-mode mit`: use direct MIT impedance control with `--mit-kp`, `--mit-kd`, `--can-port-left`, and `--can-port-right`.
- No `--async-infer`: synchronous inference; simple and easier to debug.
- `--async-infer`: background chunk prefetch; use `--trigger-step` to choose when to request the next chunk.
- `--trigger-step`: current chunk step that starts the background server request. For `--action-horizon 50`, `15` to `25` is the usual range.
- `--blend-window`: number of steps used to smooth the transition into a new chunk.
- `--record`: record MIT-control timelines under `monitor/datas/`.
- `--record-file`: output prefix for recorded timelines.

## Notes

This client targets AgileX Cobot Magic. Public package names, commands, and documentation use the AgileX Cobot Magic name.
