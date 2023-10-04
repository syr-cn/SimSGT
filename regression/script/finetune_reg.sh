#!/bin/bash
{

export name="finetune_reg";
export device="0";

python parallel_tuning_reg.py \
    --complete_feature --decay 0.002 \
    --model_file checkpoints/GEOM.pth --name $name --device $device \
    --epochs 100 --batch_size 32 --lr 0.0001\
    --eval_metric rmse --trans_encoder_layer 4 --gnn_dropout 0.5 \
    --custom_trans --transformer_norm_input --gnn_type gin_v3;
exit
}