# BreakBias

Training and evaluation code, with **U-Net Explorer** (bias-exposing view,
density-descending ADP) and **V-Net Exploiter** (anatomy-preserving view,
density-ascending ADP). 


## Environment

Linux, Python 3.10 and one CUDA GPU. The reference environment is PyTorch
2.7.1/CUDA 12.8, torchvision 0.22.1 and NumPy 1.26.4. Install compatible
CUDA-enabled PyTorch/torchvision wheels and then:

```bash
python -m pip install -r requirements.txt
```

NumPy must remain below 2 for imgaug. Higra implements the 26-connected component
tree. No SAM checkpoint, interactive model or custom C++ tree extension is used.
Different hardware/software versions are not promised bitwise identical results.

## Train

Run from the repository root. Choose the recipe explicitly:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
  train.py --config configs/la5_f02.json --seed 42 \
  --data-root /path/to/LA --output-dir outputs/f02_seed42
```

## Test best branches


```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --run-dir outputs/f02_seed42 --data-root /path/to/LA \
  --output-dir outputs/f02_seed42_test
```
