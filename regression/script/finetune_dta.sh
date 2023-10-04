#!/bin/bash
{
export name="finetune_dta";
export device="0";

python parallel_tuning_dta.py \
    --complete_feature \
    --model_file checkpoints/GEOM.pth --name $name --device $device \
    --prot_emb_dim 128 --prot_output_dim 128 --mlp_dim1 1024 --mlp_dim2 256 \
    --epochs 1000 --num_seeds 3 --lr 0.0001 --batch_size 128 --decay 0.000 \
    --trans_encoder_layer 4 --gnn_dropout 0.5 --custom_trans \
    --transformer_norm_input --gnn_type gin_v3
}