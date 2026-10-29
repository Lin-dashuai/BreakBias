# BreakBias: H03 FULL on LA V10

Minimal training and testing code for the confirmed **H03 FULL** LA 5% recipe, plus its Explorer checkpoint and the separately identified original FULL checkpoint used for the paper's historical LA5 result. No datasets, predictions, experiment logs, HPO runners, or other methods are included.

The historical `ddfp_*`, `model1/model2`, and `image_similar/image_dissimilar` interfaces are intentionally retained pending a separate terminology review. In this H03 recipe, **model1 is the UNet3D Exploiter** (anatomy-preserving view, density ascent) and **model2 is the VNet Explorer** (bias-exposing view, density descent). RealNVP/FrEIA is a density estimator, not the input-view RVP module.

## Environment

Python 3.10, Linux, and one CUDA GPU. The checked environment uses PyTorch 2.7.1/CUDA 12.8 and torchvision 0.22.1. Install the matching CUDA-enabled torch/torchvision wheels, then:

```bash
python -m pip install -r requirements.txt
```

NumPy is pinned below 2 because the retained imgaug pipeline requires its older API. Pillow/OpenCV and other dependencies required transitively by imgaug are resolved by pip. SAM, interactive-click models, a C++ tree extension, and experiment-management utilities are not required. The actual H03 tree operator uses Higra with 26-connectivity.

## Data and V10 Lists

Use the existing preprocessed LA HDF5 dataset without copying it into this repository:

```text
DATA_ROOT/CASE_ID/mri_norm2.h5   # image and label datasets
```

The supplied lists have **80 training / 10 validation / 10 test** cases. Label ratios refer to the 80-case training set:

| Ratio | Directory | Labeled | Unlabeled |
|---|---|---:|---:|
| 5% | partitions/LA/1_20 | 4 | 76 |
| 10% | partitions/LA/1_10 | 8 | 72 |
| 20% | partitions/LA/1_5 | 16 | 64 |

The training, validation and test sets are disjoint. The validation/test split was retrospectively made from the historical 20-case evaluation set with split seed 20261017; this is **not a newly unseen blind test**. The 10%/20% lists are supplied as requested, but this repository only supplies and verifies an H03 **5% training configuration**, not tuned 10%/20% recipes.

## H03 FULL Training

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  train.py --config configs/la5_h03_full.json \
  --data-root /path/to/LA --output-dir outputs/la5_h03_full
```

This is fresh training with seed 1337: UNet3D/VNet, 16 base filters, BatchNorm/dropout, FP32 training, 4 labeled plus 4 unlabeled samples per microbatch, gradient accumulation 2, SGD LR 0.01/momentum 0.9/weight decay 0.0001, the original 6000-update polynomial schedule, and full validation of both branches every 250 updates. The supervised loss is the original binary CE plus the original two-class averaged DiceLoss; it is not silently changed to a different Dice definition.

H03 uses shared tree reference, trigger probability 0.7, altitude jitter 7.5/255, ADP epsilon 0.01, **ADP loss mixture 0.7 / consistency maximum 1.0 / sigmoid ramp 1500 updates**. Its NF starts at **epoch 10**, not update 10 or a newly invented fixed-update threshold. The original EMA-parameter updates, teacher forward modes, density-estimator fitting, bidirectional main/auxiliary pseudo-supervision, and discarded final incomplete accumulation group are preserved. Auxiliary perturbed predictions are not pseudo-label sources. The legacy area/topology-probability config fields are compatibility fields: the active Higra hard filter removes propagated child-count-one nodes and does not use those fields as thresholds.

The supplied H03 ramp is 1500, whereas the manuscript's generic implementation paragraph says 2000. This release records the actual H03 run rather than changing parameters to match that paragraph.

Configuration uses standard JSON to preserve numeric types exactly, including scientific-notation NF noise values. Scientific settings are checked against the frozen H03 run; the original NumPy NF sampling/indexing logic is retained.

`pair_best.pth` and `pair_best_model1.pth`/`pair_best_model2.pth` are selected only by the highest mean of the two validation Dice scores, earliest tie. `last.pth` includes models, density estimators, optimizers, and global RNG states, but DataLoader worker/prefetch states are not captured; bitwise resume is not claimed, and this entry point intentionally starts fresh. Output directories must not already exist.

For a finite installation check, add `--smoke` and use a new output directory. It activates ADP immediately, performs two updates and validates one complete validation case. Its outputs and scores are smoke-only, not experimental results.

## Checkpoints and Evaluation Protocols

| Checkpoint | Architecture | Training crop | Test patch | Test stride XY / Z | V10 test10 raw Dice | V10 test10 LCC Dice |
|---|---|---|---|---|---:|---:|
| H03 FULL model2, validation-pair iter 2000 | VNet Explorer | 112 x 112 x 80 | 112 x 112 x 80 | 18 / 4 | 88.814262% | 88.845869% |
| Original FULL BEST model1, historical paper model | UNet3D | 96 x 96 x 96 | 96 x 96 x 96 | 16 / 16 | 88.922515% | 88.924821% |

**Stride describes validation/test sliding windows, not training stride.** These two checkpoints use the same V10 test10 cases but different inference windows, so this is not a fully matched-protocol comparison. Both retain batch 8 and their original autocast inference helpers. The original FULL also gives raw 88.808720% / LCC 88.981525% on the historical **test20**; do not label 88.98% as a V10 test10 result.

The H03 checkpoint's iteration was chosen by validation-pair mean. Historically model1 was the predefined primary; H03 model1 test10 Dice is raw 87.891163% / LCC 88.015114%. The user selected model2 for this release after examining completed test results. Its reported score must not be represented as a newly unbiased, predeclared primary result. The original FULL historical checkpoint also had test-set exposure. These historical results are supplied for transparent reproducibility, not as a new blinded model-selection claim.

Only plain inference state_dict files are shipped, not optimizer or full training-state payloads. Exact file SHA256s and foreground metrics are in `weights/manifest.json`. Training H03 does not reproduce the original FULL recipe merely because its original weight is also provided.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --preset h03-full \
  --data-root /path/to/LA --output-dir outputs/h03_test10

CUDA_VISIBLE_DEVICES=0 python test.py --preset legacy-full \
  --data-root /path/to/LA --output-dir outputs/legacy_test10
```

The script does not select a checkpoint using test results. To evaluate your own H03 model2 export, use `--checkpoint outputs/la5_h03_full/pair_best_model2.pth`. Each run preserves raw and original largest-connected-component predictions as HDF5, plus per-case **background and foreground** Dice, Jaccard, precision, recall, HD95, directed prediction-to-GT ASD, and symmetric ASSD in CSV/JSON. Surface distances are **voxels**, not verified millimeters; empty masks have explicit surface status and undefined distances. The original largest-component convention uses 3D full connectivity and its original tie rule.

## Source Acknowledgment

The component-tree and semi-supervised segmentation lineage builds on DSAIF, while the density-estimation/feature-perturbation lineage builds on DDFP and FrEIA. UNet3D, VNet, and retained density-estimation code keep their original registered parameter names for checkpoint compatibility. Renaming paper modules does not change or erase this lineage. This minimal release does not assert a new blanket license over third-party implementations.
