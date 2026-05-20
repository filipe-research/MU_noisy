# Relatório — Adição de NegGrad e SAP ao MU_noisy

**Data:** 2026-05-20
**Branch:** main
**Objetivo:** estender a comparação do paper SIBGRAPI 2026 (MU para correção de ruído) adicionando **NegGrad** e **SAP** aos métodos já existentes (SalUn, RL, FT).

---

## Estado entregue

- ✅ **Fase 1** — Mapeamento completo do repo, sem edição
- ✅ **Fase 2** — `unlearn/SAP.py` implementado + dispatcher + args
- ⏸️ **Fase 3** — Teste end-to-end **bloqueado**: nenhum baseline `exp_*_baseline_*` existe no disco
- ✅ **Fase 4** — Scripts `run_neggrad_all.sh` e `run_sap_all.sh` gerados

NegGrad já existia no repo como `--unlearn GA` (arquivo `unlearn/GA.py`); só foi orquestrado nos scripts.

---

## Fase 1 — Mapeamento do repositório

### Estrutura confirmada
Layout flat (sem subpasta `Classification/`). `unlearn/` contém: `FT.py`, `FT_prune.py`, `FT_prune_bi.py`, `GA.py`, `GA_prune.py`, `GA_prune_bi.py`, `RL.py`, `RL_pro.py`, `Wfisher.py`, `__init__.py`, `boundary_ex.py`, `boundary_sh.py`, `fisher.py`, `impl.py`, `retrain.py`.

> Note: `Wfisher.py` com W maiúsculo, mas o dispatcher mapeia `"wfisher"` → `Wfisher`.

### Padrões de assinatura
Existem **dois padrões** no `unlearn/`:

1. **Iterativos** (FT, GA, RL, retrain, wfisher, FT_prune…) usam `@iterative_unlearn` (definido em `unlearn/impl.py:54`):
   ```python
   @iterative_unlearn
   def FT(data_loaders, model, criterion, optimizer, epoch, args, mask=None): ...
   ```
   O decorador injeta `optim.SGD`, `MultiStepLR` e o loop de epochs. A assinatura externa final é `(data_loaders, model, criterion, args, mask=None, **kwargs)`.

2. **Não-iterativo** (`raw` placeholder em `unlearn/__init__.py:18`):
   ```python
   def raw(data_loaders, model, criterion, args, mask=None): pass
   ```

`main_forget.py:165` chama: `unlearn_method(unlearn_data_loaders, model, criterion, args)` — **sem `mask=`**. Logo SAP é não-iterativa, assinatura `def SAP(data_loaders, model, criterion, args)`.

### Dispatcher
`unlearn/__init__.py:22` — cadeia `if/elif name == "..."`. Para registrar SAP bastou adicionar `elif name == "SAP": return SAP`.

### Schema dos JSONs de ruído

- `cifar10_0.2_sym.json` — chaves `{noise_labels, closed_noise, clean_idx}`
  - `noise_labels`: lista de 50 000 inteiros (labels finais, já com flips)
  - `closed_noise`: lista de índices corrompidos da train set (10 000 para 20%)
  - `clean_idx`: 40 000 índices clean
- `cifar10_0.3_0.15_sym.json` (open-set) — chaves `{noise_labels, closed_noise, open_noise, clean_idx}`
  - `open_noise`: lista de pares `[idx, idx]`

`utils.setup_model_dataset()` lê esses JSONs **somente se `args.indexes_to_replace is not None`**. O comando do Filipe passa `--indexes_to_replace []` (lista vazia, truthy em Python via argparse), ativando o caminho de leitura do JSON.

### Convenção de marcação de noisy
`replace_indexes` (`dataset.py:1388`) faz `dataset.targets[indexes] = -dataset.targets[indexes] - 1` quando `only_mark=True`. No `marked_loader` os índices noisy têm target **negativo**, e `main_forget.py:67` recupera com `-original_marked_targets[mask] - 1`. ✓

### Checkpoint format
`main_forget.py:143-159` tenta as chaves `"state_dict"` → `"model"` → fallback direto, com strip de prefixo `module.`. Compatível com qualquer formato; SAP não precisa fazer nada extra.

### RTE
**Não há timer dedicado.** `FT_iter` (FT.py:31) mede tempo por batch, `_iterative_unlearn_impl` (impl.py:103) mede tempo por epoch, mas nada loga "RTE total". SAP mede explicitamente com `time.time()` e imprime no log.

### `validate()`
`trainer/val.py:6` — assinatura `validate(loader, model, criterion, args)`. Chamado automaticamente por `main_forget.py`. SAP não precisa invocar.

### Modelo
`models/ResNet.py:303` — o `_forward_impl` faz `self.normalize(x)` **antes** do `conv1`. Logo os forward_pre_hooks que SAP usa capturam ativações **já normalizadas** (comportamento desejado).

### Estado dos baselines no disco
**Nenhum `exp_*_baseline_*` existe** em `/home/pesquisador/pesquisa/` (confirmado por `find -maxdepth 4 -type d -name "exp_cifar10_nr0.2_baseline*"`). Isso bloqueia a Fase 3.

---

## Fase 2 — Implementação do SAP

### Algoritmo (versão training-free, retain-only)

1. **Identificar trusted set** dentro do `retain_loader`:
   - Forward pass, calcular CE loss por amostra
   - Estratificar por classe: pegar as `n_trusted/num_classes` amostras de **menor** loss em cada classe
   - Retornar tensor `[N_trusted, 3, H, W]`

2. **Selecionar layers alvo**:
   - Todas as `nn.Conv2d` e `nn.Linear` do modelo
   - **Exceto** a última `nn.Linear` (classifier `fc`) — consistente com `proj_classifier=False` do código oficial
   - BatchNorm e ReLU ignoradas por construção

3. **Coletar ativações de entrada** via `forward_pre_hook`:
   - Conv2d: `F.unfold(x, kernel, dilation, padding, stride)` → `[N·Hout·Wout, Cin·kh·kw]`
   - Linear: `x.reshape(-1, x.shape[-1])` → `[N·prefix, Fin]`
   - Cap por layer (`max_samples_per_layer=50000`) com subsample aleatório para não estourar RAM

4. **SVD + Eq. 7 (Kodge 2025)**:
   - SVD de `A.T` (shape `[F, N]`): `U, S, _ = torch.linalg.svd(A.T, full_matrices=False)`. `U` é a base no espaço de features.
   - `sval_ratio = S² / Σ S²`
   - `importance = α · sval_ratio / ((α − 1) · sval_ratio + 1)`
   - `P = U @ diag(importance) @ U.T` (shape `[F, F]`)

5. **Aplicar projeção nos pesos**:
   - Conv2d: `W_flat = W.view(Cout, -1); W_new = W_flat @ P.T; W = W_new.view_as(W_orig)`
   - Linear: `W = W @ P.T`
   - Bias intocado (a projeção opera só sobre o espaço de entrada)
   - Tudo dentro de `torch.no_grad()`

### Validação numérica (smoke test sem GPU)

| α | trace(P) | max eigenvalue |
|---|---|---|
| 0.5 | 0.510 | 0.025 |
| 1.0 | 1.000 | 0.049 |
| 2.0 | 1.926 | 0.093 |
| 5.0 | 4.339 | 0.204 |

Em α=1.0, `importance ≡ sval_ratio` (Σ = 1, cada termo ≤ 1) — confere com a Eq. 7. P é simétrica até precisão float (erro ~2e−9). Para resnet18, 20 layers alvo identificadas corretamente (16 BasicBlock convs + 1 conv1 + 3 downsample), `fc` **excluída** ✓.

### Decisões de design (e onde divergi do oficial)

| Decisão | Versão oficial | Versão MU_noisy | Motivo |
|---|---|---|---|
| Conjunto trusted | retain inteiro | **low-loss estratificado por classe** | Briefing pediu explicitamente. Mais robusto a ruído residual no retain e a desbalanço de classe. |
| Projeção | `I − (Mf − Mi)` (canonical) | **Mr-only** (`U·diag(imp)·U.T`) | Training-free, sem segundo hiperparâmetro (`alpha_forget`), não precisa de SVD do forget. Retém o subespaço dos clean. |
| Captura de ativações | método `get_activations()` injetado em cada layer/block | **`forward_pre_hook` em Conv2d/Linear** | Não-invasivo; funciona com qualquer arch do repo (ResNet/VGG/etc) sem reescrever modelos. |
| Camada classifier | `proj_classifier=False` (default) | **fc final pulada** | Idêntico ao default oficial. Evita destruir a calibração das logits. |
| Args nominais | `--scale_coff`, `--retain_samples` | **`--sap_alpha`, `--sap_n_trusted`** | `args.alpha` já é usado no repo para regularização l1 — não pode ser reutilizado. Nomes prefixados evitam colisão. |
| Bias | intocado (em `proj_classifier=False`) | intocado | Idêntico. |

---

## Diffs dos arquivos modificados

### `arg_parser.py`

```diff
--- a/arg_parser.py
+++ b/arg_parser.py
@@ -152,6 +152,20 @@ def parse_args():
         "--mask_path", default=None, type=str, help="the path of saliency map"
     )

+    ### SAP (Singular Value Adjusted Projection) — Kodge 2025 AAAI
+    parser.add_argument(
+        "--sap_alpha",
+        default=1.0,
+        type=float,
+        help="SAP scaling coefficient (Eq. 7 in Kodge 2025). Não usa args.alpha (já tomado).",
+    )
+    parser.add_argument(
+        "--sap_n_trusted",
+        default=1000,
+        type=int,
+        help="Número de amostras trusted (low-loss estratificado por classe) usadas pelo SAP",
+    )
+
     ### Noise Settings
```

### `unlearn/__init__.py`

```diff
--- a/unlearn/__init__.py
+++ b/unlearn/__init__.py
@@ -13,6 +13,7 @@ from .GA_prune import GA_prune
 from .RL_pro import RL_proximal
 from .boundary_ex import boundary_expanding
 from .boundary_sh import boundary_shrink
+from .SAP import SAP


 def raw(data_loaders, model, criterion, args, mask=None):
@@ -57,5 +58,7 @@ def get_unlearn_method(name):
         return boundary_shrink
     elif name == "RL_proximal":
         return RL_proximal
+    elif name == "SAP":
+        return SAP
     else:
         raise NotImplementedError(f"Unlearn method {name} not implemented!")
```

### `unlearn/SAP.py` (novo, 266 linhas)

Estrutura:

```
SAP.py
├── _identify_trusted_samples(retain_loader, model, num_classes, n_trusted, device)
│       Forward pass com CE loss; estratificação por classe; retorna (imgs, labels).
├── _select_target_layers(model)
│       Lista [(name, layer)] de Conv2d + Linear, exceto a última Linear.
├── _collect_pre_activations(model, trusted_imgs, target_layers, device, ...)
│       Registra forward_pre_hook em cada layer alvo; aplica F.unfold para Conv;
│       acumula em CPU; subsamplea se passar do cap.
├── _compute_scaled_projection(activation_mat, alpha, device)
│       SVD em [F, N]; calcula importance via Eq. 7; retorna P = U·diag(imp)·U.T.
├── _apply_projection(layer, P)
│       W_new = W_flat @ P.T (Conv: reshape; Linear: direto). Bias intocado.
└── SAP(data_loaders, model, criterion, args)  ← entry point chamado pelo dispatcher
        1) trusted = _identify_trusted_samples(...)
        2) target_layers = _select_target_layers(model)
        3) activations = _collect_pre_activations(...)
        4) para cada layer: SVD + Eq. 7 + projeção
        Mede RTE com time.time(); imprime log estruturado; retorna None.
```

---

## Fase 3 — Pendente (baseline ausente)

Nenhum diretório `exp_*_baseline_*` existe em `/home/pesquisador/pesquisa/`. Por orientação direta na sessão (resposta às perguntas), parei e estou avisando em vez de tentar treinar o baseline ou usar um checkpoint alheio.

### Comandos de teste a rodar quando o baseline existir

```bash
# CIFAR-10 sym 20%, 3 valores de α para sensibilidade
for ALPHA in 0.5 1.0 2.0; do
  python3 main_forget.py --unlearn SAP --dataset cifar10 --noise_rate 0.2 \
    --model_path exp_cifar10_nr0.2_baseline_200ep_run1/0model_SA_best.pth.tar \
    --save_dir exp_cifar10_nr0.2_sap_alpha${ALPHA} \
    --data /home/pesquisador/pesquisa/datasets \
    --sap_alpha ${ALPHA} --sap_n_trusted 1000 \
    --indexes_to_replace [] --train_seed 10 --seed 10 \
    > exp_cifar10_nr0.2_sap_alpha${ALPHA}.txt 2>&1
done
```

### Critérios de sucesso esperados

- Roda sem crash (todas as etapas `[SAP] (1/4)` … `(4/4)` aparecem)
- Salva checkpoint via `unlearn.save_unlearn_checkpoint` (linha em main_forget.py)
- Loga UA / RA / TA / SVC_MIA_forget_efficacy
- RTE: segundos a 1-2 min (não 10+ min — se for mais que isso, o `n_trusted` ou o cap de samples estão grandes demais)
- TA resultante: **~80-92%** para CIFAR-10 sym 20%
  - Se vier ~10% → SAP destruiu o modelo (α grande demais, ou problema no _apply_projection)
  - Se vier ~92%+ → SAP não fez nada (α perto de 0, ou problema na seleção trusted)
- α deve produzir variação **suave** na TA, não saltos de 50 pontos

---

## Fase 4 — Scripts de batch

### Arquivos criados (executáveis, syntax-validados com `bash -n`)

- `run_neggrad_all.sh` (168 linhas) — GA em 20 cenários × 5 seeds + Food×1 = **96 jobs**
- `run_sap_all.sh` (163 linhas) — SAP nos mesmos 20 cenários × 5 seeds + Food×1 = **96 jobs**

### Estrutura comum dos scripts

- `LOGDIR=logs`, `MASTER=logs/master_{neggrad|sap}.log`
- `DATA=/home/pesquisador/pesquisa/datasets`
- `timestamp()`, `run_cmd(name, output_file, ...)`, `check_prerequisites()`
- Helper `run_{ga|sap}_5seeds(cfg_name, dataset, noise_rate, extra_args...)`
- Logs individuais: `exp_{name}.txt` na raiz
- Convenção de seed: `run{N}` → `seed = train_seed = 10·N`

### Cenários cobertos

| Cenário | `--dataset` | `--noise_rate` | Extras | Seeds |
|---|---|---|---|---|
| CIFAR-10 sym 0.2 / 0.5 / 0.8 | `cifar10` | 0.2 / 0.5 / 0.8 | — | 5 |
| CIFAR-10 asym 40 | `cifar10` | 0.4 | `--noise_mode asym` | 5 |
| CIFAR-100 sym 0.2 / 0.5 / 0.8 | `cifar100` | 0.2 / 0.5 / 0.8 | `--noise_mode sym --num_classes 100` | 5 |
| IDN-CIFAR-10 0.2 / 0.3 / 0.4 / 0.5 | `cifar10_idn` | 0.2 / 0.3 / 0.4 / 0.5 | `--noise_mode sym` | 5 |
| IDN-CIFAR-100 0.2 / 0.3 / 0.4 / 0.5 | `cifar100_idn` | 0.2 / 0.3 / 0.4 / 0.5 | `--noise_mode sym --num_classes 100` | 5 |
| Open 15/15 | `cifar10_open` | 0.3 | `--open_ratio 0.5` | 5 |
| Open 0/30 | `cifar10_open` | 0.3 | `--open_ratio 1.0` | 5 |
| Open 30/30 | `cifar10_open` | 0.6 | `--open_ratio 0.5` | 5 |
| Open 0/60 | `cifar10_open` | 0.6 | `--open_ratio 1.0` | 5 |
| Food-101N | `food101n` | (omitido) | `--batch_size 64 --num_classes 101` (+ `--unlearn_lr 0.0013` no GA) | 1 |

### Diferenças GA vs SAP nos scripts

- **GA** passa `--unlearn_epochs 10 --unlearn_lr 0.013` (Food: 0.0013)
- **SAP** passa `--sap_alpha 1.0 --sap_n_trusted 1000` (training-free; sem epochs/lr)

---

## Recomendações de próximos passos

### Antes de rodar os batches

1. **Treine ou copie os baselines** para `exp_<cfg>_baseline_200ep_run1/0model_SA_best.pth.tar` (e `exp_food101n/0model_SA_best.pth.tar`)
2. **Confira o naming dos diretórios open/closed**: o script assume `exp_cifar10_open_closed0.15_open0.15_baseline_200ep_run1`. Se você usa o naming abreviado (ex: `exp_cifar10_open_0.15_0.15`), edite a array `CIFAR_CFGS` nos dois scripts
3. **Rode a Fase 3 manualmente** (3 valores de α) e confirme que TA está na faixa esperada antes de disparar os 96 jobs
4. Se quiser ajustar `--sap_alpha` global por dataset (CIFAR-100 pode precisar de α diferente de CIFAR-10), edite as vars `SAP_ALPHA`/`SAP_N_TRUSTED` em `run_sap_all.sh` por bloco

### Auditorias de robustez sugeridas (após o paper passar)

1. **Sensibilidade a `n_trusted`**: 500 / 1000 / 2000 / 5000 — ver se a TA é estável
2. **Sensibilidade a α**: grid mais fino em torno do best (ex: 0.7, 0.85, 1.0, 1.15, 1.3)
3. **Ablation**: comparar `trusted=retain_inteiro` vs `low-loss estratificado` — quantifica o ganho da estratificação
4. **Hooks vs reset**: rodar SAP duas vezes consecutivas — a segunda vez não deveria mudar nada (idempotência da projeção). Se mudar, há bug.

### Riscos a monitorar

- **Memória de host**: o cap `max_samples_per_layer=50000` é defensivo. Se a CPU ficar com swap em CIFAR-100 (100 classes × 50K = 5M tensors), reduza para 30000.
- **Layers downsample do ResNet**: têm kernel 1×1, `F.unfold` produz `Hout·Wout` patches por imagem com `Cin` features cada (rank ≤ Cin). SVD é estável mas pode ser barato demais — verifique no log se a projeção das `downsample.0` está fazendo algo.
- **BatchNorm em modo `eval()`**: SAP chama `model.eval()` no _identify_trusted_samples e no _collect_pre_activations. Isso usa o `running_mean/var` em vez do batch. Se você quiser que SAP modifique BN também (não é o padrão do paper), seria preciso outra rotina — não recomendado.

---

## Resumo executivo

| Item | Status |
|---|---|
| `unlearn/SAP.py` criado e auto-testado | ✅ |
| Dispatcher reconhece `--unlearn SAP` | ✅ |
| Args `--sap_alpha` e `--sap_n_trusted` registrados | ✅ |
| Métodos existentes (FT/GA/RL/wfisher/retrain/FT_prune) intocados | ✅ |
| Convenção PT nos comentários preservada | ✅ |
| Scripts batch para 96+96 jobs criados, executáveis e syntax-OK | ✅ |
| Teste end-to-end (Fase 3) | ⏸️ bloqueado por baseline ausente |
| Commits / push | ⛔ não realizado (instrução explícita do briefing) |

A próxima ação do usuário é treinar (ou copiar) o baseline `exp_cifar10_nr0.2_baseline_200ep_run1/0model_SA_best.pth.tar` e rodar o teste de sanidade de 3 α antes de disparar os scripts de batch.
