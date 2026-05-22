#!/bin/bash
DATA=/home/pesquisador/pesquisa/datasets
mkdir -p logs

train_baseline() {
    local cfg_name="$1"  # ex: cifar10_nr0.5
    local dataset="$2"
    shift 2
    local extra="$*"
    local save="exp_${cfg_name}_baseline_200ep_run1"
    if [ -f "${save}/0model_SA_best.pth.tar" ]; then
        echo "✓ ${cfg_name} já treinado, pulando"
        return
    fi
    echo "▶ ${cfg_name}"
    python3 main_train.py --arch resnet18 --dataset ${dataset} \
        --lr 0.1 --epochs 200 ${extra} \
        --data ${DATA} --save_dir ${save} \
        --indexes_to_replace [] --train_seed 10 --seed 10 \
        > logs/${save}.log 2>&1
    echo "✓ ${cfg_name} concluído"
}

# CIFAR-10 sym
train_baseline cifar10_nr0.5 cifar10 --noise_rate 0.5
train_baseline cifar10_nr0.8 cifar10 --noise_rate 0.8

# CIFAR-10 asym
train_baseline cifar10_nr0.4 cifar10 --noise_rate 0.4 --noise_mode asym

# CIFAR-100 sym
train_baseline cifar100_nr0.2 cifar100 --noise_rate 0.2 --num_classes 100
train_baseline cifar100_nr0.5 cifar100 --noise_rate 0.5 --num_classes 100
train_baseline cifar100_nr0.8 cifar100 --noise_rate 0.8 --num_classes 100

# IDN-CIFAR-10
train_baseline cifar10_idn_nr0.2 cifar10_idn --noise_rate 0.2 --noise_mode sym
train_baseline cifar10_idn_nr0.3 cifar10_idn --noise_rate 0.3 --noise_mode sym
train_baseline cifar10_idn_nr0.4 cifar10_idn --noise_rate 0.4 --noise_mode sym
train_baseline cifar10_idn_nr0.5 cifar10_idn --noise_rate 0.5 --noise_mode sym

# IDN-CIFAR-100
train_baseline cifar100_idn_nr0.2 cifar100_idn --noise_rate 0.2 --noise_mode sym --num_classes 100
train_baseline cifar100_idn_nr0.3 cifar100_idn --noise_rate 0.3 --noise_mode sym --num_classes 100
train_baseline cifar100_idn_nr0.4 cifar100_idn --noise_rate 0.4 --noise_mode sym --num_classes 100
train_baseline cifar100_idn_nr0.5 cifar100_idn --noise_rate 0.5 --noise_mode sym --num_classes 100

# (Open e Food deixados pra depois — open tem bug de marking pendente, food precisa download)

echo ""
echo "===== TODOS BASELINES TREINADOS ====="
