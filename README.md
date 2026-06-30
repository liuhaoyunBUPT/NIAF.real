# Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models

This repository contains the real-robot training, server-side, and client-side deployment code for **Neural Implicit Action Fields (NIAF)**, (ICML 2026).

The real-robot platform used here is **AgileX Cobot Magic**. Robot-side libraries, CAN setup, ROS topics, camera drivers, and the Piper SDK bindings are specific to this platform. The server and client are adapted from OpenPI's WebSocket policy server/client runtime, with additional support for NIAF-family checkpoints, AgileX Cobot Magic I/O, MIT impedance control, and asynchronous inference.

The codebase is split into three runtime roles:

| Role | Environment | Entry point | Purpose |
|---|---|---|---|
| Training | NIAF / PyTorch Lightning conda environment | `train.py` | Train NIAF, BEAST, FAST, and OFT variants on real-robot HDF5 datasets. |
| Server | OpenPI-compatible conda environment | `server.py` | Load a NIAF checkpoint and expose an OpenPI-compatible WebSocket policy server. |
| Client | uv environment on the Cobot Magic upper computer | `client/agilex_cobot_magic/main.py` | Drive the AgileX Cobot Magic robot with position or MIT impedance control. |

The simulation benchmark implementation is maintained separately at [NIAF.sim](https://github.com/liuhaoyunBUPT/NIAF.sim).

## 1. Clone

```bash
git clone --recursive git@github.com:liuhaoyunBUPT/NIAF.real.git
cd NIAF.real
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## 2. Install Environment

This repository keeps separate environments for training, policy serving, and robot execution. The server and client are both adapted from OpenPI, but they run in different places: the server runs on the policy/GPU machine in a conda environment, while the client runs on the Cobot Magic upper computer in a uv environment.

### Training Environment

```bash
conda create -n niaf_train python=3.11 -y
conda activate niaf_train
pip install -r requirements/train.txt
pip install -e .
```

`requirements/train.txt` was curated from the local validation environment used during development.

### Server Environment

```bash
conda create -n niaf_server python=3.11 -y
conda activate niaf_server
pip install -r requirements/server.txt
pip install -e third_party/openpi/packages/openpi-client
pip install -e third_party/openpi --no-deps
pip install -e .
```

`requirements/server.txt` was curated from the local validation environment used during development. `third_party/openpi` is installed with `--no-deps` because this server only uses OpenPI's serving/client protocol layer.

### Client Environment

Run this section on the **AgileX Cobot Magic upper computer**. The client follows OpenPI's real-robot example style: create a local uv virtual environment, sync the robot-client requirements, then install the OpenPI client package into that environment.

If the whole `NIAF.real` repository is available on the upper computer:

```bash
cd /path/to/NIAF.real
uv venv --python 3.10 client/agilex_cobot_magic/.venv
source client/agilex_cobot_magic/.venv/bin/activate
uv pip sync client/agilex_cobot_magic/requirements.txt
uv pip install -e third_party/openpi/packages/openpi-client
export PYTHONPATH=$PWD/client:$PWD/third_party/openpi/packages/openpi-client/src:$PYTHONPATH
```

If only the client folder is copied to the upper computer, keep the copied `packages/openpi-client` next to it and install that package instead:

```bash
cd /path/to/agilex_cobot_magic
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip sync requirements.txt
uv pip install -e packages/openpi-client
export PYTHONPATH=$PWD:$PWD/packages/openpi-client/src:$PYTHONPATH
```

The upper computer must also provide ROS, camera drivers, `cv_bridge`, socket-CAN setup, and the AgileX Piper SDK (`piper_sdk`). These are platform prerequisites and are not fully represented by pip requirements.

## 3. Repository Layout

```text
.
|-- train.py                         # Hydra/PyTorch Lightning training entry point
|-- server.py                        # Unified OpenPI WebSocket policy server
|-- configs/                         # Training configs and action statistics
|-- src/                             # Dataset, model, serving, callback, and utility code
|-- client/agilex_cobot_magic/       # Real-robot client for AgileX Cobot Magic
|-- third_party/openpi/              # Official OpenPI submodule
|-- requirements/                    # Training and server requirement files
+-- tools/                           # Preprocessing, checkpoint, and server-test utilities
```

## 4. Prepare Real-Robot Data

Training expects HDF5 episodes named `episode_*.hdf5`.

| Key | Shape | Description |
|---|---:|---|
| `action` | `(T, 14)` | Joint action sequence. |
| `observations/qpos` | `(T, 14)` | Robot joint positions. |
| `observations/images/cam_high` | `(T, H, W, 3)` | Static RGB camera. |
| `observations/images/cam_left_wrist` | `(T, H, W, 3)` | Left wrist RGB camera. |
| `observations/images/cam_right_wrist` | `(T, H, W, 3)` | Right wrist RGB camera. |
| `observations/qvel` | `(T, 14)` | Optional joint velocity sequence for velocity-supervised variants. |

Camera and state/action keys are configured in `configs/datamodule/default.yaml` and `configs/datamodule/default_vel.yaml`.

Before training on a new dataset, compute action statistics and save them under `configs/action_stats/`:

```bash
python tools/preprocess/compute_stats.py \
  --data_dir /path/to/hdf5_episodes \
  --mode all \
  --range 0.1 99.9 \
  --chunk_size 50 \
  --output configs/action_stats/my_task.yaml
```

## 5. Training

| Option | Values |
|---|---|
| Available models | NIAF (`config_niaf`), NIAF with Velocity (`config_niaf_vel`), BEAST (`config_beast`), BEAST with Velocity (`config_beast_vel`), FAST (`config_fast`), OFT (`config_oft`) |
| Default backbone | Florence-2 Large |
| Default data format | Real-robot HDF5 episodes |

Example NIAF training command:

```bash
conda activate niaf_train
PYTHONPATH=$PWD python train.py \
  --config-name config_niaf \
  root_data_dir=/path/to/hdf5_episodes \
  vlm_path=/path/to/Florence-2-large \
  checkpoint_path=null \
  trainer.devices=1 \
  num_workers=4
```

## 6. Server Deployment

Start a NIAFVel policy server:

```bash
conda activate niaf_server
PYTHONPATH=$PWD:$PWD/third_party/openpi/src:$PWD/third_party/openpi/packages/openpi-client/src \
python server.py \
  --model niaf_vel \
  --checkpoint /path/to/checkpoint.ckpt \
  --vlm-path /path/to/Florence-2-large \
  --host 0.0.0.0 \
  --port 8000 \
  --return-joint-vel \
  --no-cam-right-wrist
```

Run a local request against a running server:

```bash
conda activate niaf_server
PYTHONPATH=$PWD:$PWD/third_party/openpi/packages/openpi-client/src \
python tools/simulate_server_request.py \
  --host 127.0.0.1 \
  --port 8000 \
  --no-cam-right-wrist \
  --height 224 \
  --width 224 \
  --image-mode zeros \
  --image-range uint8
```

The response should include `actions`. If the server metadata has `returns_velocity=True`, the response also includes `velocities`, which can be consumed by the MIT impedance-control client path.

## 7. Real-Robot Client

The client is installed and run on the **AgileX Cobot Magic upper computer**. It supports two independent mode switches:

| Control mode | Inference mode | Main option |
|---|---|---|
| Position control | Synchronous inference | `--control-mode position` |
| Position control | Asynchronous inference | `--control-mode position --async-infer` |
| MIT impedance control | Synchronous inference | `--control-mode mit` |
| MIT impedance control | Asynchronous inference | `--control-mode mit --async-infer` |

Recommended real-run command after server and robot-side dependencies are validated:

```bash
cd /path/to/NIAF.real
source client/agilex_cobot_magic/.venv/bin/activate
PYTHONPATH=$PWD/client:$PWD/third_party/openpi/packages/openpi-client/src \
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

See [client/agilex_cobot_magic/README.md](client/agilex_cobot_magic/README.md) for all four launch commands and mode explanations.


## Citation

```bibtex
@article{liu2026neural,
  title={Neural Implicit Action Fields: From Discrete Waypoints to Continuous Functions for Vision-Language-Action Models},
  author={Liu, Haoyun and Zhao, Jianzhuang and Chang, Xinyuan and Shi, Tianle and Meng, Chuanzhang and Tan, Jiayuan and Xiong, Feng and Lin, Tong and Huo, Dongjie and Xu, Mu and others},
  journal={arXiv preprint arXiv:2603.01766},
  year={2026}
}
```

## License

NIAF.real is released under the MIT License. Third-party submodules, OpenPI components, robot SDKs, and vendored packages retain their original licenses.

## Acknowledgement

The server/client protocol is built around OpenPI's WebSocket policy interface. The real-robot client targets AgileX Cobot Magic and includes robot-side support for position control, MIT impedance control, and asynchronous inference.
