#!/bin/bash
set -u
DATA=/home/pesquisador/pesquisa/datasets
mkdir -p logs
run_rl() {
    local cfg="$1" dataset="$2" nr="$3"; shift 3; local extra="$*"
    local mp="exp_${cfg}_baseline_200ep_run1/0model_SA_best.pth.tar"
    local save="exp_RL_${cfg}_run1"
    [ -f "$mp" ] || { echo "X ${cfg}: sem baseline"; return; }
    [ -f "${save}/RLeval_result.pth.tar" ] && { echo "OK RL ${cfg}"; return; }
    echo ">> RL ${cfg}"
    python3 main_random.py --dataset ${dataset} --noise_rate ${nr} ${extra} \
        --unlearn_epochs 10 --unlearn RL --unlearn_lr 0.013 \
        --data ${DATA} --save_dir ${save} --model_path ${mp} \
        --indexes_to_replace [] --train_seed 10 --seed 10 \
        > logs/${save}.log 2>&1
    grep "TA (Test" logs/${save}.log | tail -1
}
run_salun() {
    local cfg="$1" dataset="$2" nr="$3"; shift 3; local extra="$*"
    local mp="exp_${cfg}_baseline_200ep_run1/0model_SA_best.pth.tar"
    local save="exp_SalUn_${cfg}_run1"
    [ -f "$mp" ] || { echo "X ${cfg}: sem baseline"; return; }
    [ -f "${save}/RLeval_result.pth.tar" ] && { echo "OK SalUn ${cfg}"; return; }
    echo ">> SalUn ${cfg}"
    python3 generate_mask.py --dataset ${dataset} --noise_rate ${nr} ${extra} \
        --unlearn_epochs 1 --data ${DATA} --save_dir ${save} \
        --model_path ${mp} --indexes_to_replace [] --train_seed 10 --seed 10 \
        >> logs/${save}.log 2>&1
    python3 main_random.py --dataset ${dataset} --noise_rate ${nr} ${extra} \
        --unlearn_epochs 10 --unlearn RL --unlearn_lr 0.013 \
        --data ${DATA} --save_dir ${save} --model_path ${mp} \
        --mask_path ${save}/with_0.5.pt \
        --indexes_to_replace [] --train_seed 10 --seed 10 \
        >> logs/${save}.log 2>&1
    grep "TA (Test" logs/${save}.log | tail -1
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
echo "===== RL ====="
for c in "${CONFIGS[@]}"; do run_rl $c; done
echo "===== SalUn ====="
for c in "${CONFIGS[@]}"; do run_salun $c; done
echo "===== FIM ====="
