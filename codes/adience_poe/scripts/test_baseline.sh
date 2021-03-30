#!/bin/bash
gpu='0'

main_loss_type='cls'
num_output_neurons=8

# main_loss_type='rank'
# num_output_neurons=16

# main_loss_type='reg'
# num_output_neurons=1


if [[ $# = 1 ]]; then
    gpu=${1}
fi

if [[ $# = 2 ]]; then
    gpu=${1}
    main_loss_type=${2}
    if [[ $main_loss_type = 'cls' ]]; then
        num_output_neurons=8
    fi
    if [[ $main_loss_type = 'rank' ]]; then
        num_output_neurons=16
    fi
    if [[ $main_loss_type = 'reg' ]]; then
        num_output_neurons=1
    fi
fi

for fold in $(seq 0 1 4); do
CUDA_VISIBLE_DEVICES=${gpu} python -u test.py \
--batch-size=32 \
--test-batch-size=32 \
--max-epochs=10 --lr-decay-epoch=7 --lr=0.0001 --fc-lr=0.0001 \
--num-output-neurons=${num_output_neurons} --main-loss-type=${main_loss_type} \
--save-model='./Save_Model' \
--train-images-root='/home/share_data/huangxiaoke/datasets/adience_dataset/aligned' \
--test-images-root='/home/share_data/huangxiaoke/datasets/adience_dataset/aligned' \
--train-data-file=./data_list/test_fold_is_${fold}/age_train.txt \
--test-data-file=./data_list/test_fold_is_${fold}/age_test.txt \
--num-workers=4 --distance='JDistance' \
--alpha-coeff=1e-5 --beta-coeff=1e-4 --margin=5 \
--exp-name=train_val_fold_${fold} \
--logdir=./log/adience_baseline_${main_loss_type} \
--no-sto \
> ./log/adience_baseline_${main_loss_type}_test_fold_${fold}.txt 2>&1
done