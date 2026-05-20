#!/bin/bash
# =============================================================================
# NegGrad (GA) — todas as configs × 5 seeds (Food-101N: 1 seed)
#
# GA já existe no repo (unlearn/GA.py) — este script só dispara via main_forget.
# Convenção de seed: run{N} -> seed = train_seed = 10 * N.
# Logs: master em logs/master_neggrad.log; por job em exp_*.txt na raiz.
# Pré-requisito: cada baseline `exp_*_baseline_200ep_run1/0model_SA_best.pth.tar`
# (e exp_food101n/0model_SA_best.pth.tar para Food) deve existir.
# =============================================================================

set -u
LOGDIR="logs"
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master_neggrad.log"

DATA="/home/pesquisador/pesquisa/datasets"
UNLEARN_EPOCHS=10
UNLEARN_LR=0.013        # CIFAR-{10,100}. Para Food-101N: 0.0013.
SEEDS=(10 20 30 40 50)  # 5 seeds para CIFAR; Food-101N usa só 10.

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Configs para checagem de pré-requisitos.
# CADA item: nome_baseline (sem o sufixo _baseline_200ep_runN). Ajuste se seus
# diretórios estiverem com nomes diferentes (ver aviso no briefing).
CIFAR_CFGS=(
    "cifar10_nr0.2"
    "cifar10_nr0.5"
    "cifar10_nr0.8"
    "cifar10_nr0.4"          # asym 40
    "cifar100_nr0.2"
    "cifar100_nr0.5"
    "cifar100_nr0.8"
    "cifar10_idn_nr0.2"
    "cifar10_idn_nr0.3"
    "cifar10_idn_nr0.4"
    "cifar10_idn_nr0.5"
    "cifar100_idn_nr0.2"
    "cifar100_idn_nr0.3"
    "cifar100_idn_nr0.4"
    "cifar100_idn_nr0.5"
    # Open/closed (closed_X_open_Y é o naming "completo"). Caso seus dirs estejam
    # encurtados (ex: exp_cifar10_open_0.15_0.15), edite as variáveis OPEN_* abaixo.
    "cifar10_open_closed0.15_open0.15"
    "cifar10_open_closed0.0_open0.3"
    "cifar10_open_closed0.3_open0.3"
    "cifar10_open_closed0.0_open0.6"
)
FOOD_BASELINE="exp_food101n/0model_SA_best.pth.tar"

check_prerequisites() {
    echo "🔍 Verificando pré-requisitos (baselines run1)..."
    local missing=0
    for cfg in "${CIFAR_CFGS[@]}"; do
        local path="exp_${cfg}_baseline_200ep_run1/0model_SA_best.pth.tar"
        if [ ! -f "$path" ]; then
            echo "   ❌ Baseline ausente: $path"
            missing=$((missing + 1))
        fi
    done
    if [ ! -f "$FOOD_BASELINE" ]; then
        echo "   ❌ Baseline Food-101N ausente: $FOOD_BASELINE"
        missing=$((missing + 1))
    fi
    if [ $missing -gt 0 ]; then
        echo "   ⚠️  Faltam $missing baseline(s). Ajuste o naming ou treine-os antes."
        return 1
    fi
    echo "   ✅ Todos os baselines encontrados"
    return 0
}

run_cmd() {
    local name="$1"
    local output_file="$2"
    shift 2
    local cmd="$*"
    echo "$(timestamp) ▶️  $name" | tee -a "$MASTER"
    eval "$cmd" > "$output_file" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "$(timestamp) ✅ $name" | tee -a "$MASTER"
    else
        echo "$(timestamp) ❌ $name (rc=$rc)" | tee -a "$MASTER"
        tail -10 "$output_file" | sed 's/^/      /' | tee -a "$MASTER"
    fi
}

# Helper genérico (CIFAR-10/100 sym/asym/idn/open). Não cobre Food-101N (que tem
# args próprios — ver bloco específico abaixo).
run_ga_5seeds() {
    local cfg_name="$1"      # ex: cifar10_nr0.2
    local dataset="$2"       # ex: cifar10
    local noise_rate="$3"    # ex: 0.2
    shift 3
    local extra_args="$*"    # ex: "--noise_mode asym" ou "--open_ratio 0.5"
    local baseline="exp_${cfg_name}_baseline_200ep_run1/0model_SA_best.pth.tar"
    for run in 1 2 3 4 5; do
        local seed=$((run * 10))
        local name="${cfg_name}_ga_run${run}"
        local save_dir="exp_${name}"
        run_cmd "$name" "exp_${name}.txt" \
            "python3 main_forget.py --unlearn GA \
             --dataset ${dataset} --noise_rate ${noise_rate} ${extra_args} \
             --model_path ${baseline} \
             --save_dir ${save_dir} \
             --data ${DATA} \
             --unlearn_epochs ${UNLEARN_EPOCHS} --unlearn_lr ${UNLEARN_LR} \
             --indexes_to_replace [] --train_seed ${seed} --seed ${seed}"
    done
}

# -----------------------------------------------------------------------------
# Início
# -----------------------------------------------------------------------------
echo "$(timestamp) ===== NegGrad batch start =====" | tee -a "$MASTER"
if ! check_prerequisites; then
    echo "$(timestamp) ❌ Pré-requisitos ausentes — abortando." | tee -a "$MASTER"
    exit 1
fi

# ----- CIFAR-10 sym -----
run_ga_5seeds "cifar10_nr0.2" "cifar10" "0.2"
run_ga_5seeds "cifar10_nr0.5" "cifar10" "0.5"
run_ga_5seeds "cifar10_nr0.8" "cifar10" "0.8"

# ----- CIFAR-10 asym 40 -----
run_ga_5seeds "cifar10_nr0.4" "cifar10" "0.4" "--noise_mode asym"

# ----- CIFAR-100 sym -----
run_ga_5seeds "cifar100_nr0.2" "cifar100" "0.2" "--noise_mode sym --num_classes 100"
run_ga_5seeds "cifar100_nr0.5" "cifar100" "0.5" "--noise_mode sym --num_classes 100"
run_ga_5seeds "cifar100_nr0.8" "cifar100" "0.8" "--noise_mode sym --num_classes 100"

# ----- IDN-CIFAR-10 (sym é o noise_mode; IDN ativa via dataset name) -----
run_ga_5seeds "cifar10_idn_nr0.2" "cifar10_idn" "0.2" "--noise_mode sym"
run_ga_5seeds "cifar10_idn_nr0.3" "cifar10_idn" "0.3" "--noise_mode sym"
run_ga_5seeds "cifar10_idn_nr0.4" "cifar10_idn" "0.4" "--noise_mode sym"
run_ga_5seeds "cifar10_idn_nr0.5" "cifar10_idn" "0.5" "--noise_mode sym"

# ----- IDN-CIFAR-100 -----
run_ga_5seeds "cifar100_idn_nr0.2" "cifar100_idn" "0.2" "--noise_mode sym --num_classes 100"
run_ga_5seeds "cifar100_idn_nr0.3" "cifar100_idn" "0.3" "--noise_mode sym --num_classes 100"
run_ga_5seeds "cifar100_idn_nr0.4" "cifar100_idn" "0.4" "--noise_mode sym --num_classes 100"
run_ga_5seeds "cifar100_idn_nr0.5" "cifar100_idn" "0.5" "--noise_mode sym --num_classes 100"

# ----- Open/closed (cifar10_open) -----
# Mapa: 15/15 -> nr=0.3 open=0.5 | 0/30 -> nr=0.3 open=1.0 | 30/30 -> nr=0.6 open=0.5 | 0/60 -> nr=0.6 open=1.0
run_ga_5seeds "cifar10_open_closed0.15_open0.15" "cifar10_open" "0.3" "--open_ratio 0.5"
run_ga_5seeds "cifar10_open_closed0.0_open0.3"   "cifar10_open" "0.3" "--open_ratio 1.0"
run_ga_5seeds "cifar10_open_closed0.3_open0.3"   "cifar10_open" "0.6" "--open_ratio 0.5"
run_ga_5seeds "cifar10_open_closed0.0_open0.6"   "cifar10_open" "0.6" "--open_ratio 1.0"

# ----- Food-101N (1 seed, batch_size=64, lr=0.0013) -----
{
    local_seed=10
    name="food101n_ga_run1"
    run_cmd "$name" "exp_${name}.txt" \
        "python3 main_forget.py --unlearn GA --dataset food101n \
         --model_path ${FOOD_BASELINE} \
         --save_dir exp_${name} \
         --data ${DATA} \
         --unlearn_epochs ${UNLEARN_EPOCHS} --unlearn_lr 0.0013 --batch_size 64 --num_classes 101 \
         --indexes_to_replace [] --train_seed ${local_seed} --seed ${local_seed}"
}

echo "$(timestamp) ===== NegGrad batch end =====" | tee -a "$MASTER"
