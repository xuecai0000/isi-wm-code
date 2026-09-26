# ISI-WM

Code for ISI-WM.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11, CUDA-enabled GPU.

## Layout

```
isi_wm/                    training and evaluation code
  common/                    interventional objective definitions
  trainer/                   training loop
  envs/                      environment wrappers and background compositor
  tools/                     evaluation and data construction scripts
  isi_wm.py                  agent (encoder, latent dynamics, MPC planner)
  config.yaml                base configuration
configs/                    intervention hyperparameters
tools/                      standalone evaluation protocol script
data_protocol/              background pool splits and compositor spec
```

## Training

```bash
cd isi_wm
python train.py task=cup-catch obs=rgb steps=300000 \
    video_background_enabled=true video_background_split=train seed=6
```

Tasks: `acrobot-swingup`, `cartpole-swingup`, `cup-catch`, `finger-spin`,
`reacher-easy`, `walker-walk`. Seeds 6, 7, 8.

## Evaluation

```bash
python -m isi_wm.tools.evaluate_cutie_multitask_checkpoint \
    --task cup-catch --backend rgb --condition hard --background-split test \
    --training-condition hard \
    --runtime-config <run>/runtime_config.json \
    --checkpoint <run>/models/final.pt \
    --training-seed 6 --env-seed 424243 --background-seed 1618034 \
    --episodes 20 --output out.json
```

## Data

Background pool splits: `data_protocol/split_*.json`. Compositor spec: `data_protocol/README.md`.
