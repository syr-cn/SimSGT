# README

## Dependencies & Dataset

PyTorch = 1.11.0

PyG == 2.0.4

You may refer to https://github.com/chao1224/GraphMVP/tree/main/datasets to download datasets.

## Reproducing results
To reproduce the results, you need to create a directory for recording results:
```
mkdir results
```

### Regression Tasks

The results on regression datasets can be reproduced using the following code.

```bash
# pretrain the model with GEOM dataset
sh script/pretrain_GEOM.sh
```

```bash
# fine-tune on the drug target affinity prediction datasets
sh script/finetune_dta.sh
python read_results.py --path results/finetune_dta

# fine-tune on the molecular property prediction datasets
sh script/finetune_reg.sh
python read_results.py --path results/finetune_reg
```

### Other tokenizers

We also tried other types of tokenziers, like BRICS tokenizer or FG tokenizers. You can reproduce these results with the following code.

Pre-training and fine-tuning with BRICS tokenizer (used by MGSSL).
```bash
sh script/run_brics.sh
python read_results.py --path results/run_brics
```

Pre-training and fine-tuning with FG tokenizer (used by RelMole).
```bash
sh script/run_fg.sh
python read_results.py --path results/run_fg
```

Pre-training and fine-tuning with pretrained tokenizer (GraphCL). Please replace LAYER_NUM with the number of layers you want to use.
```bash
export tk_layers=LAYER_NUM
sh script/run_gcl_layer$tk_layers.sh
python read_results.py --path results/run_gcl_layer$tk_layers
```

Pre-training and fine-tuning with pretrained tokenizer (VQ-VAE).
```bash
export tk_layers=LAYER_NUM
sh script/run_vae_layer$tk_layers.sh
python read_results.py --path results/run_vae_layer$tk_layers
```

Pre-training and fine-tuning with pretrained tokenizer (GraphMAE).
```bash
export tk_layers=LAYER_NUM
sh script/run_mae_layer$tk_layers.sh
python read_results.py --path results/run_mae_layer$tk_layers
```