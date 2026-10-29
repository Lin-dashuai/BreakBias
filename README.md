# BreakBias:

Training and testing code for the **BreakBias** LA 5% recipe, plus its Explorer checkpoint and the separately identified original FULL checkpoint used for the paper's historical LA5 result.

**H03 FULL uses the LA dataset protocol-version training setup: a 112 x 112 x 80 training crop.** The original FULL checkpoint corresponds to the paper's historical 96 x 96 x 96 training setup. Validation/test sliding-window strides are specified independently below: H03 XY/Z = 18/4, original FULL = 16/16.


## Environment

Python 3.10, Linux, and one CUDA GPU. The checked environment uses PyTorch 2.7.1/CUDA 12.8 and torchvision 0.22.1. Install the matching CUDA-enabled torch/torchvision wheels, then:

```bash
python -m pip install -r requirements.txt
```

NumPy is pinned below 2 because the retained imgaug pipeline requires its older API. Pillow/OpenCV and other dependencies required transitively by imgaug are resolved by pip. SAM, interactive-click models, a C++ tree extension, and experiment-management utilities are not required. The actual H03 tree operator uses Higra with 26-connectivity.

## Data and V10 Lists

Use the existing preprocessed LA HDF5 dataset:

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


