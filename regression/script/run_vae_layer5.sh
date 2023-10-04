#!/bin/bash
{
export tk_layers=5;
export name="run_vae_layer$tk_layers";
export model_file="results/$name/model_100.pth";
export device="0";

python pretraining.py \
    --batch_size 2048 --name $name --device $device \
    --dataset zinc_standard_agent --epochs 100\
    --lr 0.0001 --trans_encoder_layer 4 --trans_decoder_layer 1 --mask_rate 0.35 \
    --custom_trans --transformer_norm_input --drop_mask_tokens --nonpara_tokenizer \
    --gnn_token_layer 1 --loss ce --gnn_type gin_v3 --decoder_input_norm --eps 0.5 \
    \
    --pretrained_tokenizer --no_edge_tokenizer --tokenizer_path checkpoints/tokenizer_vae_layer$tk_layers.pth\
    --tk_gnn_type gin --tk_layers $tk_layers --tk_dropout 0.0;

python parallel_tuning.py \
    --model_file $model_file --name $name --device $device \
    --batch_size 32 --epochs 100\
    --lr 0.0001 --trans_encoder_layer 4 --gnn_dropout 0.5 \
    --custom_trans --transformer_norm_input --gnn_type gin_v3;

}