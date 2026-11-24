# RL-MSNet complete research repository

This repository contains a PPO routing-controller scaffold and the source code needed to interface with the uploaded AOSE and Cross-AVSE experts. The AOSE/AVSE parameters remain frozen. No corpus, checkpoint weights, trained visual-confidence model, or RL results are included.

## Repository map

- `configs/`: expert, dataset, PPO, segment-routing, and corruption configuration templates.
- `data/`: data-preparation notes and blank manifest templates. Put the actual corpus outside version control.
- `weights/`: place the selected AOSE TorchScript and AVSE Lightning checkpoints here locally; files are ignored by Git.
- `third_party/`: selected expert source code, excluding duplicate checkpoints and training logs.
- `src/rl_msnet/`: experts, reliability estimators, policy/reward, data helpers, and evaluation helpers.
- `scripts/`: candidate precomputation, episode building, policy training, confidence training, evaluation, corruption, and latency commands.
- `outputs/`: local checkpoints, logs, tables, and metrics; ignored by Git.
- `paper/`: methods, experiment protocol, and reviewer-response planning notes.

## Install and prepare local paths

From this repository root:

```bash
python -m venv .venv
# Activate the environment, then install the project and expert dependencies:
pip install -e '.[experts]'
```

Extract the supplied expert archives into `third_party/aose/` and `third_party/cross_avse/`, or use the source copies already supplied in those folders. Copy one validation-selected AOSE model to `weights/aose/speech_enhancer.pt` and one validation-selected AVSE checkpoint to `weights/avse/`. Set `RLMSNET_DATA_ROOT` to the local dataset root. Do not select a test checkpoint by test-set performance.

## Intended experiment flow

1. Prepare speaker-disjoint manifests with 3-second audio contexts, 75 grayscale video frames, central output crops, and the inference-only state features.
2. Run the frozen experts on training/validation/test segments and cache per-action candidate outputs and metric gains.
3. Train the visual-confidence estimator using frozen AVSE visual embeddings and training-set task-utility targets; obtain uncertainty using MC dropout and assess calibration on validation/test conditions.
4. Build the RL episode arrays, train PPO across the configured seeds, and compare with fixed, heuristic, supervised, and soft-gating baselines.
5. Evaluate stitched selected outputs and report PESQ, STOI, SI-SDR, route proportions, switch rate, confidence calibration, and measured end-to-end runtime.

The current scripts implement the expert adapters, cached-utility PPO training, confidence-estimator training from prepared feature arrays, candidate precomputation, and expert latency measurement. They do **not** automatically prepare the corpus, generate confidence-estimator embeddings/targets, implement every comparison baseline, stitch final routed audio, or run statistical tests. The corresponding modules mark those integration points; do not report results until those steps are implemented and run on your data.

## Critical model constraint

The supplied AVSE model uses a 3-second context and up to 75 frames at 25 fps. It is not causal streaming AVSE. The initial protocol is segment-level: one decision per 1-second interval using the fixed 3-second context and overlap/cross-fade settings in `configs/routing_segment.yaml`. Measure repeated context processing in the deployed RTF. Do not describe this as frame-level routing.

## Running the current controller scaffold

Install the project in editable mode first. Use `python scripts/build_rl_episodes.py --help` and `python scripts/train_ppo.py --help` to see required inputs. PPO costs and reward penalties must be passed explicitly and selected using training/validation data. Example command template:

```bash
python scripts/train_ppo.py --data data/episodes/train.npz --out outputs/checkpoints \
  --seeds 11,22,33,44,55 --action-costs AOSE_COST AVSE_COST NSE_COST \
  --lambda-compute COMPUTE_WEIGHT --lambda-switch SWITCH_WEIGHT
```

Replace uppercase placeholders with measured/validation-selected numeric values. The YAML files are protocol templates; the current PPO script does not load them automatically.

## Data-to-policy commands

The intended processing order is:

```bash
python scripts/prepare_video.py path/to/clip.mp4 path/to/clip.npy
python scripts/precompute_candidates.py --manifest data/manifests/train.csv \
  --aose weights/aose/speech_enhancer.pt --avse-checkpoint weights/avse/selected.ckpt \
  --avse-source third_party/cross_avse --out outputs/candidate_metrics/train.csv
python scripts/build_confidence_dataset.py --manifest data/manifests/train.csv \
  --candidate-csv outputs/candidate_metrics/train.csv \
  --avse-checkpoint weights/avse/selected.ckpt --avse-source third_party/cross_avse \
  --out outputs/candidate_metrics/confidence_train.npz \
  --sisdr-scale-db VALUE --stoi-scale VALUE --temperature VALUE
python scripts/train_visual_confidence.py --train outputs/candidate_metrics/confidence_train.npz \
  --validation outputs/candidate_metrics/confidence_validation.npz \
  --out outputs/checkpoints/visual_confidence.pt
```

Then predict confidence for each split with `scripts/predict_visual_confidence.py`, join it with cached candidate metrics using `scripts/assemble_rl_transitions.py`, convert the joined table with `scripts/build_rl_episodes.py`, and train PPO. Build the validation and test candidate tables separately; never train the policy or tune reward weights on test metrics. Exact argument values and output paths depend on your local corpus and measured validation scales.
