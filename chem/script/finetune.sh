#!/bin/bash
{
export name="finetune";
export device="0";

python parallel_tuning.py --model_file checkpoints/m35_ckt.pt --name $name --device $device --lr 0.0001 --trans_encoder_layer 4 --gnn_dropout 0.5 --custom_trans --transformer_norm_input --gnn_type gin_v3;
exit
}