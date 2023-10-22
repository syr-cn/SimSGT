#!/bin/bash
{

export device="0";
export name="finetune";

python parallel_tuning_qm.py --clf_norm layer \
    --decay 0.000 --probe_lr 0.0001 --probe_dropout 0.2\
    --model_file checkpoints/m35_ckt.pt --name $name --device $device \
    --epochs 100 --batch_size 32 --lr 0.0001 --eval_metric mae --scheduler None\
    --eval_metric mae --trans_encoder_layer 4 --gnn_dropout 0.5 --transformer_dropout 0.0 \
    --custom_trans --transformer_norm_input --gnn_type gin_v3 \
    2>log/$name.err >log/$name.out;
}