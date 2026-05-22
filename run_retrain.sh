#!/bin/bash
set -u
DATA=/home/pesquisador/pesquisa/datasets
mkdir -p logs
run_retrain() {
    local cfg="$1" ds="$2" nr="$3"; shift 3; local extra="$*"
    local mp="exp_${cfg}_baseline_200ep_run1/0model_SA_best.pth.tar"
    local save="exp_Retrain_${cfg}_run1"
    [ -f "$mp" ] || { echo "X ${cfg}: sem baseline"; return; }
    [ -f "logs/${save}.log" ] && grep -q "best SA" "logs/${save}.log" && { echo "OK Retrain ${cfg}"; return; }
    echo ">> Retrain ${cfg}"
    python3 my_retrain.py --dataset ${ds} --noise_rate ${nr} ${extra} \
        --unlearn retrain --epochs 200 --lr 0.1 \
        --data ${DATA} --save_dir ${save} --model_path ${mp} \
        --indexes_to_replace [] --train_seed 10 --seed 10 \
        > logs/${save}.log 2>&1
    grep "best SA" logs/${save}.log | tail -1
}
CONFIGS=(
    "cifar10_nr0.2 cifar10 0.2"
    "cifar10_nr0.5 cifar10 0.5"
    "cifar10_nr0.8 cifar10 0.8"
    "cifar10_nr0.4 cifar10 0.4 --noise_mode asym"
    "cifar100_nr0.2 cifar100 0.2"
    "cifar100_nr0.5 cifar100 0.5"
    "cifar100_nr0.8 cifar100 0.8"
    "cifar10_idn_nr0.2 cifar10_idn 0.2 --noise_mode sym"
    "cifar10_idn_nr0.3 cifar10_idn 0.3 --noise_mode sym"
    "cifar10_idn_nr0.4 cifar10_idn 0.4 --noise_mode sym"
    "cifar10_idn_nr0.5 cifar10_idn 0.5 --noise_mode sym"
    "cifar100_idn_nr0.2 cifar100_idn 0.2 --noise_mode sym"
    "cifar100_idn_nr0.3 cifar100_idn 0.3 --noise_mode sym"
    "cifar100_idn_nr0.4 cifar100_idn 0.4 --noise_mode sym"
    "cifar100_idn_nr0.5 cifar100_idn 0.5 --noise_mode sym"
)
echo "===== RETRAIN (15 CIFAR, 200ep) ====="
for c in "${CONFIGS[@]}"; do run_retrain $c; done
echo "===== FIM ====="
