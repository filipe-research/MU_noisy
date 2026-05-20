"""
SAP — Singular Value Adjusted Projection
(Kodge, Ravikumar, Saha, Roy — AAAI 2025, https://arxiv.org/abs/2403.08618)

Implementação training-free: cirurgia pós-hoc nos pesos do modelo via
SVD das ativações de um subconjunto "trusted" (presumivelmente clean).
Não usa SGD/Adam, não tem epochs nem learning rate.

Adaptações ao pipeline MU_noisy (ver README do briefing):
  - Trusted set = low-loss estratificado por classe DENTRO do retain_loader
    (n_trusted/num_classes amostras por classe, escolhidas por menor CE loss).
  - Projeção retain-only: P = U @ diag(importance) @ U.T, onde U vem do
    SVD das ativações de entrada (pre-activation) e `importance` segue a
    Eq. 7 do paper: alpha * sval_ratio / ((alpha-1)*sval_ratio + 1).
  - Camadas tratadas: Conv2d e Linear, EXCETO a última Linear (classifier).
    BatchNorm é ignorada.
  - Ativações coletadas via forward_pre_hook (não invasivo — funciona com
    qualquer arquitetura do repo sem precisar reescrever os modelos).
"""

import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def _identify_trusted_samples(retain_loader, model, num_classes, n_trusted, device):
    """Seleciona n_trusted/num_classes amostras de menor CE loss por classe.

    Retorna lista [(image_tensor, label_int)] no host (cpu) — vamos repassar
    em batches depois.
    """
    model.eval()
    ce_none = nn.CrossEntropyLoss(reduction="none")

    # Buffers por classe — separados em listas Python pra evitar overhead inicial
    per_class_losses = [[] for _ in range(num_classes)]
    per_class_imgs = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for image, target in retain_loader:
            image = image.to(device, non_blocking=True)
            target_d = target.to(device, non_blocking=True)
            output = model(image)
            losses = ce_none(output, target_d).detach().cpu()
            # Mover imagens pra cpu uma única vez
            image_cpu = image.detach().cpu()
            tgt_cpu = target.detach().cpu()
            for i in range(image_cpu.size(0)):
                c = int(tgt_cpu[i].item())
                if 0 <= c < num_classes:
                    per_class_losses[c].append(float(losses[i].item()))
                    per_class_imgs[c].append(image_cpu[i])

    quota = max(1, n_trusted // max(num_classes, 1))
    trusted_imgs = []
    trusted_labels = []
    for c in range(num_classes):
        if not per_class_imgs[c]:
            continue
        order = np.argsort(np.array(per_class_losses[c]))[:quota]
        for idx in order:
            trusted_imgs.append(per_class_imgs[c][int(idx)])
            trusted_labels.append(c)

    # Stack final pra um único tensor (host)
    if len(trusted_imgs) == 0:
        raise RuntimeError("[SAP] Nenhuma amostra trusted encontrada — retain_loader vazio?")
    return torch.stack(trusted_imgs, dim=0), torch.tensor(trusted_labels, dtype=torch.long)


def _select_target_layers(model):
    """Retorna [(name, module)] de todas Conv2d/Linear, removendo a última Linear.

    A última Linear é assumida como classifier (fc final no ResNet/VGG do repo)
    e pulada para preservar a calibração das logits, seguindo o
    proj_classifier=False default do código oficial.
    """
    cand = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            cand.append((name, layer))
    # Localiza o índice da última Linear
    last_linear = None
    for i in reversed(range(len(cand))):
        if isinstance(cand[i][1], nn.Linear):
            last_linear = i
            break
    if last_linear is not None:
        # Pula o classifier final
        cand = cand[:last_linear] + cand[last_linear + 1:]
    return cand


def _collect_pre_activations(model, trusted_imgs, target_layers, device,
                             batch_size=128, max_samples_per_layer=50000):
    """Roda forward das amostras trusted com forward_pre_hooks, capturando
    a matriz de entradas de cada Conv2d/Linear-alvo.

    Para Conv2d: aplica F.unfold para virar [N*Hout*Wout, Cin*kh*kw].
    Para Linear: faz reshape pra [N, Fin] (achatando dims de prefixo, se houver).

    Subsamplea por layer se o total acumulado ultrapassa max_samples_per_layer
    (evita matrizes gigantes — Conv inicial em CIFAR já gera 1024 linhas por
    imagem; em batches grandes, sem cap, explode).
    """
    activations = OrderedDict()
    handles = []

    def make_hook(layer_name):
        def hook(module, inputs):
            x = inputs[0].detach()
            if isinstance(module, nn.Conv2d):
                # [N, Cin, H, W] -> [N, Cin*kh*kw, L] -> [N*L, Cin*kh*kw]
                unfolded = nn.functional.unfold(
                    x,
                    kernel_size=module.kernel_size,
                    dilation=module.dilation,
                    padding=module.padding,
                    stride=module.stride,
                )
                mat = unfolded.permute(0, 2, 1).contiguous().view(-1, unfolded.shape[1])
            elif isinstance(module, nn.Linear):
                # Achata dims do prefixo (caso o input tenha mais de 2 dims)
                mat = x.reshape(-1, x.shape[-1])
            else:
                return
            mat = mat.cpu()
            if layer_name in activations:
                activations[layer_name] = torch.cat([activations[layer_name], mat], dim=0)
            else:
                activations[layer_name] = mat
            # Cap pra não estourar memória de host
            cur = activations[layer_name]
            if cur.size(0) > max_samples_per_layer:
                idx = torch.randperm(cur.size(0))[:max_samples_per_layer]
                activations[layer_name] = cur[idx].contiguous()
        return hook

    for name, layer in target_layers:
        h = layer.register_forward_pre_hook(make_hook(name))
        handles.append(h)

    model.eval()
    try:
        with torch.no_grad():
            for batch in torch.split(trusted_imgs, batch_size):
                _ = model(batch.to(device, non_blocking=True))
    finally:
        # Garante remoção dos hooks mesmo se algo falhar
        for h in handles:
            h.remove()

    return activations


def _compute_scaled_projection(activation_mat, alpha, device):
    """SVD da matriz de ativações e construção da projeção SAP no espaço de features.

    activation_mat: tensor [N, F] (linhas = amostras/patches, colunas = features in).
    Retorna P [F, F] = U_F @ diag(importance) @ U_F.T, onde U_F são os vetores
    singulares à esquerda de [F, N] (= features x samples).

    Eq. 7 do paper:
        sval_ratio = S^2 / sum(S^2)
        importance = alpha * sval_ratio / ((alpha - 1) * sval_ratio + 1)
    """
    A = activation_mat.to(device).float()
    # SVD em [F, N] dá U como base no espaço de features (o que precisamos)
    A_t = A.t().contiguous()
    # full_matrices=False mantém U com shape [F, r] (r=min(F,N))
    U, S, _ = torch.linalg.svd(A_t, full_matrices=False)
    # Eq. 7 do paper (Kodge 2025): normalizacao pelo MAXIMO singular value,
    # nao pela soma — soma faz a projecao escalar tudo por uma fracao pequena.
    sval_max = S[0]  # S vem ordenada desc por torch.linalg.svd
    sval_norm_sq = (S / (sval_max + 1e-12)) ** 2  # em (0, 1]
    importance = alpha * sval_norm_sq / ((alpha - 1.0) * sval_norm_sq + 1.0)
    # P = U diag(importance) U^T  (sem o sqrt: queremos a projeção final, não o "feature mat")
    P = U @ torch.diag(importance) @ U.t()
    return P


def _apply_projection(layer, P):
    """Aplica W_new = W_flat @ P^T (projeção no espaço de input da camada).

    Para Conv2d: achata kernel [Cout, Cin, kh, kw] -> [Cout, Cin*kh*kw], multiplica
    pela projeção [F,F] e desfaz o reshape.
    Para Linear: multiplicação direta.
    Biases não são tocados (a projeção opera só sobre o espaço de entrada).
    """
    P_dev = P.to(layer.weight.device, dtype=layer.weight.dtype)
    if isinstance(layer, nn.Conv2d):
        W = layer.weight.data
        W_flat = W.view(W.shape[0], -1)
        W_new = W_flat @ P_dev.t()
        layer.weight.data = W_new.view_as(W)
    elif isinstance(layer, nn.Linear):
        W = layer.weight.data
        W_new = W @ P_dev.t()
        layer.weight.data = W_new


def SAP(data_loaders, model, criterion, args):
    """SAP training-free: cirurgia nos pesos do modelo via SVD das ativações trusted.

    Args necessários (definidos em arg_parser.py):
        --sap_alpha     : coef. de escala da Eq. 7 (default 1.0)
        --sap_n_trusted : nº de amostras trusted (default 1000)

    Mantém a interface do dispatcher do main_forget.py: recebe data_loaders
    (OrderedDict retain/forget/val/test), model, criterion, args. Modifica o
    modelo in-place. Não retorna nada (assim como `raw`).
    """
    device = next(model.parameters()).device
    retain_loader = data_loaders["retain"]

    print("=" * 64)
    print(f"[SAP] alpha={args.sap_alpha}  n_trusted={args.sap_n_trusted}")
    print(f"[SAP] num_classes={args.num_classes}  device={device}")
    print("=" * 64)

    t_start = time.time()

    # 1) Identificar trusted samples (low-loss estratificado por classe)
    print("[SAP] (1/4) selecionando trusted samples low-loss por classe...")
    trusted_imgs, trusted_labels = _identify_trusted_samples(
        retain_loader=retain_loader,
        model=model,
        num_classes=int(args.num_classes),
        n_trusted=int(args.sap_n_trusted),
        device=device,
    )
    print(f"[SAP]      trusted_set: {trusted_imgs.shape[0]} amostras "
          f"({trusted_imgs.shape[0] // max(int(args.num_classes), 1)} por classe alvo)")

    # 2) Identificar layers alvo (Conv + Linear, exceto a fc final)
    target_layers = _select_target_layers(model)
    print(f"[SAP] (2/4) layers alvo (Conv/Linear, exceto classifier): {len(target_layers)}")
    for name, lyr in target_layers:
        print(f"[SAP]      - {name}  ({type(lyr).__name__})")

    # 3) Coletar ativações de entrada com forward_pre_hooks
    print("[SAP] (3/4) coletando ativações pre via hooks...")
    activations = _collect_pre_activations(
        model=model,
        trusted_imgs=trusted_imgs,
        target_layers=target_layers,
        device=device,
    )

    # 4) Para cada layer: SVD, Eq. 7, projeção dos pesos
    print("[SAP] (4/4) SVD + Eq. 7 + projeção dos pesos...")
    layer_map = {name: lyr for name, lyr in target_layers}
    with torch.no_grad():
        for name, act_mat in activations.items():
            layer = layer_map[name]
            P = _compute_scaled_projection(act_mat, float(args.sap_alpha), device)
            _apply_projection(layer, P)
            print(f"[SAP]      - {name:32s}  act={tuple(act_mat.shape)}  P={tuple(P.shape)}")

    rte = time.time() - t_start
    print("=" * 64)
    print(f"[SAP] concluído. RTE = {rte:.2f} s")
    print("=" * 64)

    return None
