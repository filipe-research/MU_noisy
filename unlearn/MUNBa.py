"""MUNBa: Machine Unlearning via Nash Bargaining (Wu & Harandi, CVPR 2025).

Implements Algorithm 1 + closed-form solution (Eq. 8) from
https://arxiv.org/abs/2411.15537, following the reference implementation at
https://github.com/JingWu321/MUNBa (Classification/unlearn/MUNBa.py).

Per-iteration update:
    1) loss_r = CE(model(x_r), y_r)                # preservation player
    2) loss_u = CE(model(x_u), y_u_random)         # forgetting player
       (paper text says "maximize CE on forget"; reference code uses
       random-label CE — pure ascent diverges in practice.)
    3) g_r = grad(loss_r), g_u = grad(loss_u) (computed once for alpha only)
       g1 = ||g_r||^2, g2 = <g_r,g_u>, g3 = ||g_u||^2
       alpha_r = sqrt((g1*g3 - g2*sqrt(g1*g3)) / (g1^2*g3 - g1*g2^2 + eps))
       alpha_u = (1 - g1*alpha_r^2) / (g2*alpha_r + eps)
       If alpha_r <= 0 or alpha_u <= 0 -> fallback (1.0, 0.1).
    4) loss = alpha_r * loss_r + alpha_u * loss_u
       loss.backward(); clip_grad_norm_(1.0); optimizer.step()
"""

import sys
import time

import torch
import torch.nn as nn
import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


_EPS = 1e-8
_CLIP_NORM = 1.0
_FALLBACK_ALPHA_R = 1.0
_FALLBACK_ALPHA_U = 0.1
_FORGET_WEIGHT = 0.1


@iterative_unlearn
def MUNBa(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]

    losses_r = utils.AverageMeter()
    losses_u = utils.AverageMeter()
    top1_r = utils.AverageMeter()
    cos_meter = utils.AverageMeter()

    model.train()
    forget_iter = iter(forget_loader)

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    params = [p for p in model.parameters() if p.requires_grad]

    start = time.time()
    for i, retain_batch in enumerate(retain_loader):
        if args.imagenet_arch:
            x_r, y_r = get_x_y_from_data_dict(retain_batch, device)
        else:
            x_r, y_r = retain_batch
            x_r = x_r.cuda()
            y_r = y_r.cuda()

        if epoch < args.warmup:
            utils.warmup_lr(
                epoch, i + 1, optimizer,
                one_epoch_step=len(retain_loader), args=args,
            )

        try:
            forget_batch = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            forget_batch = next(forget_iter)

        if args.imagenet_arch:
            x_f, _ = get_x_y_from_data_dict(forget_batch, device)
        else:
            x_f, _ = forget_batch
            x_f = x_f.cuda()

        y_f_random = torch.randint(
            0, args.num_classes, (x_f.shape[0],), device=x_f.device
        )

        optimizer.zero_grad(set_to_none=True)

        output_r = model(x_r)
        loss_r = criterion(output_r, y_r)
        output_u = model(x_f)
        loss_u = criterion(output_u, y_f_random)

        grad_r_list = torch.autograd.grad(
            loss_r, params, retain_graph=True, allow_unused=False,
        )
        grad_u_list = torch.autograd.grad(
            loss_u, params, retain_graph=True, allow_unused=False,
        )
        with torch.no_grad():
            g_r = torch.cat([g.detach().reshape(-1) for g in grad_r_list])
            g_u = torch.cat([g.detach().reshape(-1) for g in grad_u_list])

            g1 = (g_r * g_r).sum()
            g3 = (g_u * g_u).sum()
            g2 = (g_r * g_u).sum()
            norm_r = torch.sqrt(g1 + _EPS)
            norm_u = torch.sqrt(g3 + _EPS)
            cos_theta = g2 / (norm_r * norm_u + _EPS)

            # Closed-form (Eq. 8 / reference code form):
            #   alpha_r = sqrt((g1*g3 - g2*sqrt(g1*g3)) / (g1^2*g3 - g1*g2^2 + eps))
            #   alpha_u = (1 - g1*alpha_r^2) / (g2*alpha_r + eps)
            denom = g1 * g1 * g3 - g1 * g2 * g2
            num = g1 * g3 - g2 * torch.sqrt(g1 * g3 + _EPS)
            ratio = num / (denom + _EPS)
            alpha_r_val = torch.sqrt(torch.clamp(ratio, min=0.0) + _EPS)
            alpha_u_val = (1.0 - g1 * alpha_r_val * alpha_r_val) / (
                g2 * alpha_r_val + _EPS
            )

            if (
                not torch.isfinite(alpha_r_val)
                or not torch.isfinite(alpha_u_val)
                or alpha_r_val.item() <= 0
                or alpha_u_val.item() <= 0
            ):
                alpha_r_val = torch.tensor(_FALLBACK_ALPHA_R, device=g_r.device)
                alpha_u_val = torch.tensor(_FALLBACK_ALPHA_U, device=g_r.device)

            # Down-weight the forgetting player. The closed-form Nash solution
            # equalises retain and forget contributions in norm; with random-
            # label forget loss this floods the update with noise (we observed
            # train_acc ~97% but eval ~50%). The reference repo's arg_parser
            # carries --lam default 0.1 with the same intent (Wu & Harandi,
            # see official repo). Forget player gets 10% bargaining weight.
            alpha_u_val = _FORGET_WEIGHT * alpha_u_val

        # Free per-loss grad lists before single combined backward.
        del grad_r_list, grad_u_list, g_r, g_u

        loss = alpha_r_val * loss_r + alpha_u_val * loss_u
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), _CLIP_NORM)

        if mask is not None:
            for name, p in model.named_parameters():
                if p.grad is not None and name in mask:
                    p.grad *= mask[name]

        optimizer.step()

        with torch.no_grad():
            prec1_r = utils.accuracy(output_r.float().data, y_r)[0]
        losses_r.update(loss_r.item(), x_r.size(0))
        losses_u.update(loss_u.item(), x_f.size(0))
        top1_r.update(prec1_r.item(), x_r.size(0))
        cos_meter.update(cos_theta.item(), 1)

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "MUNBa Epoch: [{0}][{1}/{2}]\t"
                "Loss_r {lr.val:.4f} ({lr.avg:.4f})\t"
                "Loss_u {lu.val:.4f} ({lu.avg:.4f})\t"
                "Acc_r {top1.val:.3f} ({top1.avg:.3f})\t"
                "cos {cm.val:+.3f} ({cm.avg:+.3f})\t"
                "a_r {ar:.3f} a_u {au:.3f}\t"
                "||gr|| {nr:.3f} ||gu|| {nu:.3f}\t"
                "Time {3:.2f}".format(
                    epoch, i, len(retain_loader), end - start,
                    lr=losses_r, lu=losses_u, top1=top1_r, cm=cos_meter,
                    ar=float(alpha_r_val), au=float(alpha_u_val),
                    nr=norm_r.item(), nu=norm_u.item(),
                )
            )
            start = time.time()

    print(
        "MUNBa epoch {} retain_acc {top1.avg:.3f} "
        "loss_r {lr.avg:.4f} loss_u {lu.avg:.4f} cos_avg {cm.avg:+.3f}".format(
            epoch, top1=top1_r, lr=losses_r, lu=losses_u, cm=cos_meter,
        )
    )
    return top1_r.avg
