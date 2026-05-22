#!/bin/bash
DATA=/home/pesquisador/pesquisa/datasets
mkdir -p logs
train_open() {
    local name="$1" nr="$2" op="$3"
    local save="exp_cifar10_open_${name}_baseline_200ep_run1"
    [ -f "${save}/0model_SA_best.pth.tar" ] && { echo "OK ${name} ja existe"; return; }
    echo ">> treinando ${name} (nr=${nr} op=${op})"
    python3 main_train.py --arch resnet18 --dataset cifar10_open --lr 0.1 --epochs 200 \
        --noise_rate ${nr} --open_ratio ${op} \
        --data ${DATA} --save_dir ${save} \
        --indexes_to_replace [] --train_seed 10 --seed 10 \
        > logs/${save}.log 2>&1
    grep "best SA" logs/${save}.log | tail -1
}
train_open closed0.15_open0.15 0.3 0.5
train_open closed0.0_open0.3   0.3 1.0
train_open closed0.3_open0.3   0.6 0.5
train_open closed0.0_open0.6   0.6 1.0
