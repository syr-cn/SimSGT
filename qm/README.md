# README

## Dependencies & Dataset

PyTorch = 1.11.0

PyG == 2.0.4

The quantum mechanics datasets are from [MoleculeNet](https://moleculenet.org/datasets-1).

## Reproducing results
To reproduce the results, you need to create a directory for recording results:
```bash
mkdir results
```

Then run the following code for fine-tuning on the three quantum mechanics datasets:

```bash
# fune-tune a pretrained checkpoint
sh script/finetune.sh

python read_results.py --path results/finetune/
```