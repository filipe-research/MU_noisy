#!/bin/bash
set -u
DATA=/home/pesquisador/pesquisa/datasets
mkdir -p logs

run_unlearn() {
    local method="$1"
    local cfg="$2"
    local dataset="$3"
    local nr="$4"
    local lr="$5"
    shift 5
    local extra="$*"
    
    local model_path="exp_${cfg}_baseline_200ep_run1/0model_SA_best.pth.tar"
    local save="exp_${method}_${cfg}_run1"
    local log="logs/${save}.log"
    
    if [ ! -f "$model_path" ]; then
        echo "✗ ${cfg}: baseline não existe, pulando"
        return
    fi
    if [ -f "${save}/${method}eval_result.pth.tar" ]; then
        echo "✓ ${method} ${cfg}: já rodou"
        return
    fi
    
    echo "▶ ${method} ${cfg} (lr=${lr})"
    python3 main_forget.py --unlearn ${method} \
        --dataset ${dataset} --noise_rate ${nr} ${extra} \
        --model_path ${model_path} \
        --save_dir ${save} \
        --data ${DATA} \
        --unlearn_epochs 10 --unlearn_lr ${lr} \
        --indexes_to_replace [] --train_seed 10 --seed 10 \
        > ${log} 2>&1
    
    if [ $? -ne 0 ]; then
        echo "  ✗ falhou — ver ${log}"
    else
        grep -E "TA \(" ${log} | tail -1
    fi
}

echo "=== Rodando GA (lr=1e-4) em todos baselines ==="
run_unlearn GA cifar10_nr0.2 cifar10 0.2 0.0001
run_unlearn GA cifar10_nr0.5 cifar10 0.5 0.0001
run_unlearn GA cifar10_nr0.8 cifar10 0.8 0.0001
run_unlearn GA cifar10_nr0.4 cifar10 0.4 0.0001 --noise_mode asym

echo ""
echo "=== Rodando MUNBa (lr=0.013) em todos baselines ==="
run_unlearn MUNBa cifar10_nr0.2 cifar10 0.2 0.013
run_unlearn MUNBa cifar10_nr0.5 cifar10 0.5 0.013
run_unlearn MUNBa cifar10_nr0.8 cifar10 0.8 0.013
run_unlearn MUNBa cifar10_nr0.4 cifar10 0.4 0.013 --noise_mode asym

echo ""
echo "=== Re-rodando FT pra controle ==="
run_unlearn FT cifar10_nr0.2 cifar10 0.2 0.013
run_unlearn FT cifar10_nr0.5 cifar10 0.5 0.013
run_unlearn FT cifar10_nr0.8 cifar10 0.8 0.013
run_unlearn FT cifar10_nr0.4 cifar10 0.4 0.013 --noise_mode asym

echo ""
echo "===== SUMÁRIO ====="
printf "%-30s %s\n" "Run" "TA"
echo "------------------------------------------------"
for log in logs/exp_GA_*.log logs/exp_MUNBa_*.log logs/exp_FT_*.log; do
    [ -f "$log" ] || continue
    name=$(basename "$log" .log | sed 's/^exp_//')
    ta=$(grep "TA (Test Accuracy)" "$log" | tail -1 | grep -oE "[0-9]+\.[0-9]+%")
    printf "%-30s %s\n" "${name}" "${ta:-FAIL}"
done
