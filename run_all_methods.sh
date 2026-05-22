#!/bin/bash
set -u
DATA=/home/pesquisador/pesquisa/datasets
mkdir -p logs

run_unlearn() {
    local method="$1" lr="$2" cfg="$3" dataset="$4" nr="$5"
    shift 5
    local extra="$*"
    local model_path="exp_${cfg}_baseline_200ep_run1/0model_SA_best.pth.tar"
    local save="exp_${method}_${cfg}_run1"
    local log="logs/${save}.log"
    [ -f "$model_path" ] || { echo "✗ ${cfg}: sem baseline"; return; }
    [ -f "${save}/${method}eval_result.pth.tar" ] && { echo "✓ ${method} ${cfg}: já rodou"; return; }
    echo "▶ ${method} ${cfg} (lr=${lr})"
    python3 main_forget.py --unlearn ${method} \
        --dataset ${dataset} --noise_rate ${nr} ${extra} \
        --model_path ${model_path} --save_dir ${save} --data ${DATA} \
        --unlearn_epochs 10 --unlearn_lr ${lr} \
        --indexes_to_replace [] --train_seed 10 --seed 10 \
        > ${log} 2>&1
    [ $? -eq 0 ] && grep "TA (Test" ${log} | tail -1 || echo "  ✗ falhou - ver ${log}"
}

declare -a CONFIGS=(
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

echo "===== FT ====="
for cfg in "${CONFIGS[@]}"; do run_unlearn FT 0.013 $cfg; done
echo "===== GA (NegGrad) ====="
for cfg in "${CONFIGS[@]}"; do run_unlearn GA 0.0001 $cfg; done
echo "===== MUNBa ====="
for cfg in "${CONFIGS[@]}"; do run_unlearn MUNBa 0.013 $cfg; done
