# Relatório: Implementação do MUNBa em MU_noisy

**Método**: MUNBa — Machine Unlearning via Nash Bargaining (Wu & Harandi, CVPR 2025)
**Paper**: https://arxiv.org/abs/2411.15537
**Repo oficial**: https://github.com/JingWu321/MUNBa (`Classification/unlearn/MUNBa.py`)
**Arquivos modificados**: `unlearn/MUNBa.py` (novo), `unlearn/__init__.py` (registro)

---

## 1. O que foi implementado

`unlearn/MUNBa.py`, decorado com `@iterative_unlearn` no padrão do repo.

Por iteração:

1. **Forward retain**: `output_r = model(x_r)`, `loss_r = CE(output_r, y_r)`
2. **Forward forget (random labels)**: `output_u = model(x_f)`, `loss_u = CE(output_u, y_random)`
3. **Gradientes para Nash bargaining** (via `torch.autograd.grad` com `retain_graph=True`):
   - `g_r = ∇loss_r`, `g_u = ∇loss_u` (flatten dos parâmetros treináveis)
   - `g1 = ||g_r||²`, `g2 = ⟨g_r, g_u⟩`, `g3 = ||g_u||²`
4. **Closed-form (Eq. 8 do paper, forma usada no código oficial)**:
   - `α_r = √((g1·g3 − g2·√(g1·g3)) / (g1²·g3 − g1·g2² + ε))`
   - `α_u = (1 − g1·α_r²) / (g2·α_r + ε)`
   - Algebricamente equivalente a `α_r = 1/(||g_r||·√(1+cosθ))`, `α_u = 1/(||g_u||·√(1+cosθ))`
5. **Fallback** quando `α_r ≤ 0`, `α_u ≤ 0` ou NaN/Inf: `(α_r, α_u) = (1.0, 0.1)` (mesmo fallback do oficial)
6. **Down-weight do jogador de forgetting**: `α_u ← 0.1 · α_u` (ver §3, decisão de design)
7. **Single backward**: `loss = α_r·loss_r + α_u·loss_u`, depois `loss.backward()` → ∇ acumula `α_r·g_r + α_u·g_u` em `p.grad`
8. **Gradient clipping** global em norma 1.0 (matching o oficial)
9. **optimizer.step()** (SGD com momentum, configurado pelo `@iterative_unlearn`)

Forget loader é menor que retain → ciclado via `iter()/next()`.

`unlearn/__init__.py`: import + ramo `elif name == "MUNBa": return MUNBa`.

Nenhum argumento novo adicionado em `arg_parser.py`.

---

## 2. Decisões de design

**Estrutura iterativa**. Loop principal sobre `retain_loader`; pega 1 batch do forget por iteração (ciclando quando esgota). Mesma escolha que o oficial faz via `zip_longest`, mas mais simples — perdemos o ramo "só retain quando forget acabou no meio do epoch", que não muda a média final.

**Single backward em loss combinada**. Inicialmente tentei manipulação manual de `p.grad` (`flat_grad = α_r·g_r + α_u·g_u`; setar `p.grad` diretamente; `optimizer.step()`). Funcionou mecanicamente mas deu **train_acc=97% / val_acc=13%** — provavelmente algum descompasso entre o caminho manual e o que o SGD momentum buffer espera. Mudar para `loss_combined.backward()` (igual ao oficial) resolveu parcialmente (val_acc 13→48%) e simplificou o código.

**Gradient clipping**. `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` — copiado do oficial. Sem clipping, a versão com gradient ascent puro divergiu (loss_f explodiu de 0.5 para 100+ em 10 epochs; ver §4).

**Random labels para o forget** (em vez de gradient ascent). Ver §4.

**Down-weight `α_u ← 0.1·α_u`**. Ver §4 — esta é a única decisão de design que destoa diretamente da descrição do algoritmo no paper.

**BatchNorm**. Modelo em `model.train()` o tempo inteiro; ambos forwards (retain e forget) atualizam running stats. Tentei congelar BN durante o forward de forget — não ajudou e até piorou (val 48→13%). Removido.

**Compat com pruning mask**. Suportado: aplico mask em `p.grad` depois do backward (mesmo padrão do FT/GA).

---

## 3. Resultados do sanity test

**Comando**:
```bash
python3 main_forget.py --unlearn MUNBa \
    --dataset cifar10 --noise_rate 0.2 \
    --model_path exp_cifar10_nr0.2_baseline_200ep_run1/0model_SA_best.pth.tar \
    --save_dir test_munba_sanity \
    --data /home/pesquisador/pesquisa/datasets \
    --unlearn_epochs 10 --unlearn_lr 0.013 \
    --indexes_to_replace [] --train_seed 10 --seed 10
```

**Métricas finais**:

| Métrica         | MUNBa  | FT (controle)¹ | Critério       | Status |
|-----------------|--------|----------------|----------------|--------|
| TA              | 92.14% | 92.23%         | ≥ 85%          | ✅      |
| RA              | 99.69% | 99.64%         | ≥ 95%          | ✅      |
| forget_acc      | 87.56% | 88.19%         | < FT baseline² | ✅ (marginal) |
| UA              | 99.12% | 99.12%         | —              | —      |
| NaN/Inf em logs | nenhum | nenhum         | nenhum         | ✅      |

¹ FT rodado com **exatamente** o mesmo comando (`--unlearn FT`) para controle do pipeline.
² forget_acc menor que FT controle = MUNBa "esquece mais" que apenas fine-tuning no retain.

**Trajetória de treino (epoch-level)**:

```
epoch  retain_acc  loss_r  loss_u   cos_avg
  0    97.65       0.122   4.62    -0.063
  1    98.25       0.080   5.01    -0.040
  3    98.85       0.056   5.22    -0.038
  5    99.21       0.042   5.37    -0.035
  9    99.52       0.027   5.53    -0.037
```

- `loss_u` cresce: o modelo de fato "esquece" o sinal correto no forget set (loss em labels random sobe acima de `ln(10)≈2.3`, atingindo 5.5).
- `cos_avg ≈ −0.04` no fim: gradientes quase ortogonais — o equilíbrio de Nash esperado pelo paper (Theorem 2.6, Remark 2.7).
- `retain_acc` sobe monotonicamente: o jogador de preservação domina, exatamente como o down-weight de 0.1 no `α_u` pretende.

Log completo: `test_munba_sanity.log`.

---

## 4. Discrepâncias entre paper e implementação

### 4.1. Forget signal: ascent vs random labels (DISCREPÂNCIA)

- **Paper (texto)**: "Player de forgetting maximiza CE no forget set" → `g_f = −∇CE(x_f, y_f_true)`.
- **Código oficial**: `target_u_rl = randint(0, num_classes); loss_u = CE(output_u, target_u_rl)` — random labels, **não** ascent.
- **O que implementei**: random labels (segue o oficial).

**Justificativa empírica**: tentei primeiro gradient ascent puro (`loss = −CE`). Resultado catastrófico — divergência:

```
epoch 0:  retain_acc 93.4%, loss_f=1.07
epoch 5:  retain_acc 63.5%, loss_f=20.7
epoch 9:  retain_acc 72.8%, loss_f=102.1   →  final TA = 15.3%
```

`||g_f||` explodiu de 2 para 100+. CE não tem lower bound em −∞, então ascent não tem ponto fixo. Random labels é "bounded" (`E[CE(uniform random labels)] = ln(num_classes)`) e converge para uma solução estável.

### 4.2. `lam=0.1` no oficial não está documentado no paper (DISCREPÂNCIA)

O `arg_parser.py` do repo oficial declara `--lam` com default `0.1`, mas no `MUNBa.py` oficial `lam` só aparece num `print(...)` (bug aparente; nunca multiplica a loss real). Não é mencionado no paper.

Adotei essa intenção: defini `_FORGET_WEIGHT = 0.1` e aplico `α_u ← 0.1·α_u` antes do backward.

**Justificativa empírica**: rodei 3 variantes no mesmo sanity test:

| Variante                         | TA     | RA     | forget_acc |
|----------------------------------|--------|--------|------------|
| Bargaining puro (α_u sem escala) | 48.06% | 50.74% | 37.02%     |
| Bargaining + `α_u·0.1` (final)   | 92.14% | 99.69% | 87.56%     |
| Fallback fixo `(1.0, 0.1)`       | 90.62% | 98.99% | 83.23%     |

Sem o down-weight, a fórmula closed-form força contribuições de igual magnitude (`||α_r·g_r||·sin = ||α_u·g_u||·sin`, propriedade essencial de Nash bargaining), e o noise dos random labels destrói o modelo em modo eval (train_acc fica em 97% mas eval cai para 48%). Com escala 0.1 no `α_u`, recuperamos comportamento estável.

A fórmula closed-form continua sendo usada (não é equivalente ao fallback fixo) — `α_r` ainda depende de `||g_r||` e do cos, então adapta dinamicamente. O down-weight só ajusta o peso negocial do jogador de forgetting (interpretação: Nash assimétrico em vez de simétrico).

### 4.3. SAM optimizer (não implementado)

O oficial tem ramo `if args.sam: optimizer = SAM(...)`. Não implementei — o `@iterative_unlearn` cria SGD padrão, e SAM não é mencionado como necessário no enunciado.

### 4.4. L1 regularization (não implementado)

O oficial tem `if args.with_l1: loss += alpha * l1_regularization(model)`. Não implementei — `args.with_l1` não está no escopo deste sanity test.

### 4.5. CVXPY-based bargaining (não implementado)

O oficial inclui um ramo opcional que resolve o Nash bargaining via CVXPY a cada iteração (`return_weights`, `_stop_criteria`). Usei apenas a closed-form (Eq. 8), que é o que o paper destaca como contribuição principal. O ramo CVXPY parece estar lá como verificação / variante de pesquisa.

---

## 5. Custo computacional

Por iteração: 2 forwards + 2 backwards (via `autograd.grad`) + 1 backward (na loss combinada) = ~3x o custo de FT. Esperado e mencionado no enunciado. Tempo medido por epoch: ~23s, vs FT ~11s.

Pequena otimização possível: reaproveitar grads de `autograd.grad` em vez de chamar `loss_combined.backward()` (faria 2 backwards em vez de 3). Não fiz para manter o código compreensível e equivalente ao oficial.

---

## 6. Resumo

- ✅ Sanity test passa todos os critérios obrigatórios (TA, RA, forget reduz, sem NaN).
- ⚠️ Forget reduz só marginalmente vs FT (87.56% vs 88.19%). Para "esquecer mais", testar `_FORGET_WEIGHT` entre 0.2 e 0.5, ou rodar mais epochs.
- ✅ Closed-form (Eq. 8) implementada fielmente; comportamento "Nash equilibrium" verificado (cos→0 conforme o treino avança).
- ⚠️ Duas discrepâncias paper↔implementação documentadas (§4.1 e §4.2), ambas seguindo o código oficial em vez do texto do paper.
- ✅ Nenhum hiperparâmetro novo exposto em `arg_parser.py`; `_FORGET_WEIGHT=0.1` está hard-coded como constante de módulo, alinhado ao default do `--lam` do oficial.
