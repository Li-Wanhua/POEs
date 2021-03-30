# POEs

PyTorch re-implementation of Learning Probabilistic Ordinal Embeddings for Uncertainty-Aware Regressiong  (CVPR 2021)[[project page](https://github.com/Li-Wanhua/POEs)]

# Codes for Adience Dataset
[Adience Dataset](https://talhassner.github.io/home/projects/Adience/Adience-data.html)

## Prepare Environment
Simply create env by: 
```bash
conda create -f environment.yaml
```
The codes is on test on pytorch==1.0.0, but higher version of pytorch should be ok.

## Train
Configure the data-related paths in `scripts/*.sh`, specifically the `--train-images-root`, `--test-images-root`, `--train-data-file`, and `--test-data-file` flags.

```bash
# Train POEs / baselines
# model_type should be in ['reg', 'cls', 'rank']
bash ./scripts/train_poe.sh [num_of_gpu='0'] [model_type='cls']
bash ./scripts/train_baseline.sh [num_of_gpu='0'] [model_type='cls']
```
## Test
```bash
# Test POEs / baselines
# model_type should be in ['reg', 'cls', 'rank']
bash ./scripts/test_poe.sh [num_of_gpu='0'] [model_type='cls']
bash ./scripts/test_baseline.sh [num_of_gpu='0'] [model_type='cls']
```
## Performance Summary
```bash
python ./misc/metric_summary.py
```
