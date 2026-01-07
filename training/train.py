import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import ClipLoss, KDClipLoss, get_cast_dtype, TaxoDistillLoss, TaxoDistillConfig
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model




#新增
def mu_schedule(step, total_steps, warm_frac=0.3, mu_max=0.2):
    """
    Distance-aware KD µÄÈ¨ÖØµ÷¶È£º
    - Ç° warm_frac µÄÑµÁ·½ø¶È£ºmu = 0£¨±ÜÃâ¹ýÔçÆ½»¬Ó°Ïì top-1£©
    - ºóÃæÏßÐÔÉýµ½ mu_max
    """
    p = step / max(1, total_steps)
    if p < warm_frac:
        return 0.0
    return mu_max * (p - warm_frac) / (1.0 - warm_frac)







def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        
        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for





# def train_kd_one_epoch(model, t_model, data, epoch, loss, optimizer, scaler, scheduler, args, tb_writer=None):
#     device = torch.device(args.device)
#     autocast = get_autocast(args.precision)
#     cast_dtype = get_cast_dtype(args.precision)
#
#     model.train()
#
#     data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
#     dataloader = data['train'].dataloader
#     num_batches_per_epoch = dataloader.num_batches
#     sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))
#
#     loss_m = AverageMeter()
#     loss_task = AverageMeter()
#     loss_icl = AverageMeter()
#     loss_ckd = AverageMeter()
#     loss_cross_kd  = AverageMeter()
#     loss_fd = AverageMeter()
#     loss_gd = AverageMeter()
#     loss_afd = AverageMeter()
#     batch_time_m = AverageMeter()
#     data_time_m = AverageMeter()
#     end = time.time()
#     for i, batch in enumerate(dataloader):
#         step = num_batches_per_epoch * epoch + i
#
#         if not args.skip_scheduler:
#             scheduler(step)
#
#         images, texts = batch
#         images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
#         texts = texts.to(device=device, non_blocking=True)
#
#         data_time_m.update(time.time() - end)
#         optimizer.zero_grad()
#
#         with autocast():
#             image_features, text_features, logit_scale = model(images, texts, distill=True, mask_ratio=args.mask_ratio)
#
#             with torch.no_grad():
#                 t_image_features, t_text_features, t_logit_scale = t_model(images, texts)
#
#             losses = loss(image_features, text_features, logit_scale, \
#                 t_image_features, t_text_features, t_logit_scale)
#
#             task_loss, ckd_loss, icl_loss, cross_kd_loss, fd_loss, gd_loss, afd_loss = losses
#             total_loss = task_loss + ckd_loss + icl_loss + cross_kd_loss + fd_loss + gd_loss + afd_loss
#
#         if scaler is not None:
#             scaler.scale(total_loss).backward()
#             if args.horovod:
#                 optimizer.synchronize()
#                 scaler.unscale_(optimizer)
#                 if args.grad_clip_norm is not None:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
#                 with optimizer.skip_synchronize():
#                     scaler.step(optimizer)
#             else:
#                 if args.grad_clip_norm is not None:
#                     scaler.unscale_(optimizer)
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
#                 scaler.step(optimizer)
#             scaler.update()
#         else:
#             total_loss.backward()
#             if args.grad_clip_norm is not None:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
#             optimizer.step()
#
#         # Note: we clamp to 4.6052 = ln(100), as in the original paper.
#         with torch.no_grad():
#             unwrap_model(model).logit_scale.clamp_(0, math.log(100))
#
#         batch_time_m.update(time.time() - end)
#         end = time.time()
#         batch_count = i + 1
#         if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
#             batch_size = len(images)
#             num_samples = batch_count * batch_size * args.world_size
#             samples_per_epoch = dataloader.num_samples
#             percent_complete = 100.0 * batch_count / num_batches_per_epoch
#
#             # NOTE loss is coarsely sampled, just master node and per log update
#             loss_m.update(total_loss.item(), batch_size)
#             loss_task.update(task_loss.item(), batch_size)
#             loss_icl.update(icl_loss.item(), batch_size)
#             loss_ckd.update(ckd_loss.item(), batch_size)
#             loss_cross_kd.update(cross_kd_loss.item(), batch_size)
#             loss_fd.update(fd_loss.item(), batch_size)
#             loss_gd.update(gd_loss.item(), batch_size)
#             loss_afd.update(afd_loss.item(), batch_size)
#             logit_scale_scalar = logit_scale.item()
#             logging.info(
#                 f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
#                 f"Total Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
#                 f"Task Loss: {loss_task.val:#.5g} ({loss_task.avg:#.4g}) "
#                 f"ICL Loss: {loss_icl.val:#.5g} ({loss_icl.avg:#.4g}) "
#                 f"CKD Loss: {loss_ckd.val:#.5g} ({loss_ckd.avg:#.4g}) "
#                 f"Cross KD Loss: {loss_cross_kd.val:#.5g} ({loss_cross_kd.avg:#.4g}) "
#                 f"FD Loss: {loss_fd.val:#.5g} ({loss_fd.avg:#.4g}) "
#                 f"GD Loss: {loss_gd.val:#.5g} ({loss_gd.avg:#.4g}) "
#                 f"AFD Loss: {loss_afd.val:#.5g} ({loss_afd.avg:#.4g}) "
#                 f"Data (t): {data_time_m.avg:.3f} "
#                 f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
#                 f"LR: {optimizer.param_groups[0]['lr']:5f} "
#                 f"Logit Scale: {logit_scale_scalar:.3f}"
#             )
#
#             # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
#             log_data = {
#                 "loss": loss_m.val,
#                 "data_time": data_time_m.val,
#                 "batch_time": batch_time_m.val,
#                 "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
#                 "scale":  logit_scale_scalar,
#                 "lr": optimizer.param_groups[0]["lr"]
#             }
#             for name, val in log_data.items():
#                 name = "train/" + name
#                 if tb_writer is not None:
#                     tb_writer.add_scalar(name, val, step)
#                 if args.wandb:
#                     assert wandb is not None, 'Please install wandb.'
#                     wandb.log({name: val, 'step': step})
#
#             # resetting batch / data time meters per log window
#             batch_time_m.reset()
#             data_time_m.reset()
#


#xinzeng
def compute_taxo_weight(epoch, i, num_batches_per_epoch, args):
    """
    ·µ»Øµ±Ç° step µÄ alpha_taxo£¨0 -> alpha_taxo_max ÏßÐÔÔö³¤£©
    ÓÃ epoch µÄÐ¡Êý½ø¶È£ºepoch + i/num_batches_per_epoch
    """
    prog = epoch + (i + 1) / float(num_batches_per_epoch)  # e.g. 2.3

    if prog < args.taxo_warmup_epochs:
        return 0.0

    # ramp from 0 to 1
    ramp_t = (prog - args.taxo_warmup_epochs) / max(args.taxo_ramp_epochs, 1e-8)
    ramp_t = max(0.0, min(1.0, ramp_t))
    return args.alpha_taxo_max * ramp_t




#新增
def train_kd_one_epoch(model, t_model, data, epoch, loss, taxo_loss_fn, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
#xinzeng
    m = model.module if hasattr(model, "module") else model
    if args.lock_text:
        m.transformer.eval()

    t_model.eval()

    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    # total steps for scheduling mu_dist across whole training
    total_steps = num_batches_per_epoch * args.epochs

    loss_m = AverageMeter()
    loss_task = AverageMeter()
    loss_icl = AverageMeter()
    loss_ckd = AverageMeter()
    loss_cross_kd = AverageMeter()
    loss_fd = AverageMeter()
    loss_gd = AverageMeter()
    loss_afd = AverageMeter()

    # ===== ADD: Taxo loss meters =====
    loss_taxo_total = AverageMeter()
    loss_taxo_hier = AverageMeter()
    loss_taxo_dist = AverageMeter()
    loss_taxo_repr = AverageMeter()

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i

        if not args.skip_scheduler:
            scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        with autocast():
            # ===== student forward =====
            image_features, text_features, logit_scale = model(images, texts, distill=True, mask_ratio=args.mask_ratio)

            # ===== teacher forward =====
            with torch.no_grad():
                t_image_features, t_text_features, t_logit_scale = t_model(images, texts)

            # ===== original KDClipLoss =====
            task_loss, ckd_loss, icl_loss, cross_kd_loss, fd_loss, gd_loss, afd_loss = \
                loss(image_features, text_features, logit_scale,
                     t_image_features, t_text_features, t_logit_scale)

            total_loss = task_loss + ckd_loss + icl_loss + cross_kd_loss + fd_loss + gd_loss + afd_loss

            alpha_taxo = compute_taxo_weight(epoch, i, num_batches_per_epoch, args)

            # ===== ADD: taxonomy distillation loss =====
            # schedule mu_dist (distance-aware KD weight)
            mu = mu_schedule(
                step=step,
                total_steps=total_steps,
                warm_frac=0.3,                       # ÄãºóÃæÒ²¿ÉÒÔ×ö³É args.taxo_warm_frac
                mu_max=float(taxo_loss_fn.cfg.mu_dist)
            )

            taxo_loss, taxo_parts = taxo_loss_fn(
                s_image_features=image_features,
                t_image_features=t_image_features,
                s_logit_scale=logit_scale,
                t_logit_scale=t_logit_scale,
                mu_dist=mu,
                return_parts=True
            )

            total_loss = total_loss + alpha_taxo *taxo_loss

        # ===== backward / step =====
        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # clamp logit_scale
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1

        # ===== logging =====
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # update meters
            loss_m.update(total_loss.item(), batch_size)
            loss_task.update(task_loss.item(), batch_size)
            loss_icl.update(icl_loss.item(), batch_size)
            loss_ckd.update(ckd_loss.item(), batch_size)
            loss_cross_kd.update(cross_kd_loss.item(), batch_size)
            loss_fd.update(fd_loss.item(), batch_size)
            loss_gd.update(gd_loss.item(), batch_size)
            loss_afd.update(afd_loss.item(), batch_size)

            # taxo parts
            loss_taxo_total.update(taxo_loss.item(), batch_size)
            # taxo_parts ÊÇ detach µÄÕÅÁ¿£¨ÔÚ TaxoDistillLoss ÀïÎÒÃÇ×öÁË detach£©
            # ÕâÀï safe È¡ .item()
            if "loss_hier" in taxo_parts:
                loss_taxo_hier.update(float(taxo_parts["loss_hier"].item()), batch_size)
            if "loss_dist" in taxo_parts:
                loss_taxo_dist.update(float(taxo_parts["loss_dist"].item()), batch_size)
            if "loss_repr" in taxo_parts:
                loss_taxo_repr.update(float(taxo_parts["loss_repr"].item()), batch_size)

            logit_scale_scalar = logit_scale.item()

            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Total: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) | "
                f"Task: {loss_task.val:#.5g} | ICL: {loss_icl.val:#.5g} | CKD: {loss_ckd.val:#.5g} | "
                f"CrossKD: {loss_cross_kd.val:#.5g} | FD: {loss_fd.val:#.5g} | GD: {loss_gd.val:#.5g} | AFD: {loss_afd.val:#.5g} | "
                f"Taxo: {loss_taxo_total.val:#.5g} (H:{loss_taxo_hier.val:#.4g}, D:{loss_taxo_dist.val:#.4g}, R:{loss_taxo_repr.val:#.4g}, mu:{mu:.3f}) | "
                f"Data(t): {data_time_m.avg:.3f} | Batch(t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s | "
                f"LR: {optimizer.param_groups[0]['lr']:5f} | LogitScale: {logit_scale_scalar:.3f}"
                f" | Taxo: {taxo_loss.item():#.5g} (alpha:{alpha_taxo:.4f}, H:{...}, D:{...}, R:{...}, mu:{...}) | ..."
            )

            # TB/W&B
            log_data = {
                "loss": loss_m.val,
                "task_loss": loss_task.val,
                "icl_loss": loss_icl.val,
                "ckd_loss": loss_ckd.val,
                "cross_kd_loss": loss_cross_kd.val,
                "fd_loss": loss_fd.val,
                "gd_loss": loss_gd.val,
                "afd_loss": loss_afd.val,

                "taxo_loss": loss_taxo_total.val,
                "taxo_hier": loss_taxo_hier.val,
                "taxo_dist": loss_taxo_dist.val,
                "taxo_repr": loss_taxo_repr.val,
                "taxo_mu": mu,

                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size * args.world_size / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
            }

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            batch_time_m.reset()
            data_time_m.reset()





def evaluate(model, data, epoch, args, tb_writer=None):
    import json
    import logging
    import os
    import torch
    import torch.nn.functional as F
    import numpy as np

    metrics = {}
    if not is_master(args):
        return metrics

    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        cumulative_loss = 0.0
        all_image_features, all_text_features, all_text_tokens = [], [], []

        # ÓÃÈ« val µÄ¾ùÖµ logit_scale£¨±È¡°×îºóÒ»¸ö batch¡±¸üÎÈ£©
        ls_sum = 0.0
        ls_cnt = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)

                    # accumulate features on CPU
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    all_text_tokens.append(texts.cpu())

                    # mean logit scale
                    ls = float(logit_scale.mean().item())
                    ls_sum += ls
                    ls_cnt += 1

                    # keep original contrastive val loss (batch-wise)
                    logits_per_image = logit_scale.mean() * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()
                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size

                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t"
                    )

            # ---- NEW metrics: grouped by unique prompt tokens ----
            mean_logit_scale = torch.tensor(ls_sum / max(1, ls_cnt))
            val_metrics = get_metrics_grouped(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                text_tokens=torch.cat(all_text_tokens),
                logit_scale=mean_logit_scale,
                k_list=(1, 5, 10),
                chunk_size=2048
            )

            loss = cumulative_loss / num_samples
            metrics.update({**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_metrics_grouped(image_features, text_features, text_tokens, logit_scale, k_list=(1, 5, 10), chunk_size=2048):
    """
    ÊÊÅä¡°Í¬Ò»¸ö species prompt ÖØ¸´ºÜ¶à´Î¡±µÄÆÀ¹À£º
    - ÓÃ text_tokens È¥ÖØµÃµ½ unique prompt Àà±ð£¨¡ÖÎïÖÖÊý£©
    - ¹¹½¨Ã¿¸öÀà±ðµÄ text prototype£¨Æ½¾ù text feature£©
    - image->text(class)£ºÏàµ±ÓÚÎïÖÖ top-k
    - text(class)->image£º¶àÕýÑù±¾ recall£¨Í¬ÀàÈÎÒâÃüÖÐËã¶Ô£©

    Args:
      image_features: [N, D] CPU Tensor
      text_features:  [N, D] CPU Tensor
      text_tokens:    [N, L] CPU LongTensor (token ids)
      logit_scale:    scalar tensor (CPU) or float
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    if not torch.is_tensor(logit_scale):
        logit_scale = torch.tensor(logit_scale)

    # normalize (±£ÏÕ)
    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)

    # unique token rows -> class ids
    # inv: [N] each sample -> class id
    _, inv = torch.unique(text_tokens, dim=0, return_inverse=True)
    inv = inv.long()
    n_classes = int(inv.max().item() + 1)

    N, D = text_features.shape

    # build text prototypes: [C, D]
    proto = torch.zeros((n_classes, D), dtype=text_features.dtype)
    proto.scatter_add_(0, inv.view(-1, 1).expand(-1, D), text_features)
    counts = torch.bincount(inv, minlength=n_classes).clamp_min(1).to(proto.dtype)
    proto = proto / counts.view(-1, 1)
    proto = F.normalize(proto, dim=-1)

    # GT class for each image
    gt = inv  # [N]

    metrics = {}
    metrics["num_images"] = N
    metrics["num_classes"] = n_classes

    # ---- image -> class top-k (species top-k) ----
    img2cls_hits = {k: 0 for k in k_list}
    maxK = max(k_list)

    # rank@maxK (½üËÆ£¬Ö»Í³¼ÆÔÚ top-maxK ÄÚµÄ rank£»Ã»ÃüÖÐÔò¼ÇÎª maxK)
    first_rank_all = []

    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        logits = (logit_scale * (image_features[s:e] @ proto.t()))  # [B, C]
        topk = torch.topk(logits, k=maxK, dim=1).indices            # [B, maxK]
        g = gt[s:e].view(-1, 1)                                     # [B, 1]

        match = (topk == g)
        first = torch.where(
            match.any(dim=1),
            match.float().argmax(dim=1),
            torch.full((e - s,), maxK, dtype=torch.long)
        )
        first_rank_all.append(first.cpu())

        for k in k_list:
            hit = (topk[:, :k] == g).any(dim=1).sum().item()
            img2cls_hits[k] += int(hit)

    first_rank_all = torch.cat(first_rank_all).numpy()
    metrics["image_to_text_mean_rank@maxK"] = float(first_rank_all.mean() + 1)
    metrics["image_to_text_median_rank@maxK"] = float(np.floor(np.median(first_rank_all)) + 1)
    for k in k_list:
        metrics[f"image_to_text_R@{k}"] = img2cls_hits[k] / N

    # ---- class -> image multi-positive recall ----
    # For each class c, positives are all images with gt==c; hit if any of top-k images matches class c.
    cls2img_hits = {k: 0 for k in k_list}
    C = n_classes
    gt_cpu = gt  # already cpu

    for cs in range(0, C, chunk_size):
        ce = min(C, cs + chunk_size)
        logits = (logit_scale * (proto[cs:ce] @ image_features.t()))  # [Bc, N]
        topk = torch.topk(logits, k=maxK, dim=1).indices              # [Bc, maxK]
        cls_ids = torch.arange(cs, ce).view(-1, 1)                    # [Bc, 1]
        topk_img_cls = gt_cpu[topk]                                   # [Bc, maxK]

        for k in k_list:
            hit = (topk_img_cls[:, :k] == cls_ids).any(dim=1).sum().item()
            cls2img_hits[k] += int(hit)

    for k in k_list:
        metrics[f"text_to_image_R@{k}"] = cls2img_hits[k] / C

    return metrics









# def evaluate(model, data, epoch, args, tb_writer=None):
#     metrics = {}
#     if not is_master(args):
#         return metrics
#     device = torch.device(args.device)
#     model.eval()
#
#     zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
#     metrics.update(zero_shot_metrics)
#
#
#     autocast = get_autocast(args.precision)
#     cast_dtype = get_cast_dtype(args.precision)
#
#     if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
#         dataloader = data['val'].dataloader
#         num_samples = 0
#         samples_per_val = dataloader.num_samples
#
#         # FIXME this does not scale past small eval datasets
#         # all_image_features @ all_text_features will blow up memory and compute very quickly
#         cumulative_loss = 0.0
#         all_image_features, all_text_features = [], []
#         with torch.no_grad():
#             for i, batch in enumerate(dataloader):
#                 images, texts = batch
#                 images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
#                 texts = texts.to(device=device, non_blocking=True)
#
#                 with autocast():
#                     image_features, text_features, logit_scale = model(images, texts)
#                     # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
#                     # however, system RAM is easily exceeded and compute time becomes problematic
#                     all_image_features.append(image_features.cpu())
#                     all_text_features.append(text_features.cpu())
#                     logit_scale = logit_scale.mean()
#                     logits_per_image = logit_scale * image_features @ text_features.t()
#                     logits_per_text = logits_per_image.t()
#
#                     batch_size = images.shape[0]
#                     labels = torch.arange(batch_size, device=device).long()
#                     total_loss = (
#                         F.cross_entropy(logits_per_image, labels) +
#                         F.cross_entropy(logits_per_text, labels)
#                     ) / 2
#
#                 cumulative_loss += total_loss * batch_size
#                 num_samples += batch_size
#                 if is_master(args) and (i % 100) == 0:
#                     logging.info(
#                         f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
#                         f"Loss: {cumulative_loss / num_samples:.6f}\t")
#
#             val_metrics = get_metrics(
#                 image_features=torch.cat(all_image_features),
#                 text_features=torch.cat(all_text_features),
#                 logit_scale=logit_scale.cpu(),
#             )
#             loss = cumulative_loss / num_samples
#             metrics.update(
#                 {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
#             )
#
#     if not metrics:
#         return metrics
#
#     logging.info(
#         f"Eval Epoch: {epoch} "
#         + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
#     )
#
#     if args.save_logs:
#         for name, val in metrics.items():
#             if tb_writer is not None:
#                 tb_writer.add_scalar(f"val/{name}", val, epoch)
#
#         with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
#             f.write(json.dumps(metrics))
#             f.write("\n")
#
#     if args.wandb:
#         assert wandb is not None, 'Please install wandb.'
#         for name, val in metrics.items():
#             wandb.log({f"val/{name}": val, 'epoch': epoch})
#
#     return metrics
#
#
# def get_metrics(image_features, text_features, logit_scale):
#     metrics = {}
#     logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
#     logits_per_text = logits_per_image.t().detach().cpu()
#
#     logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
#     ground_truth = torch.arange(len(text_features)).view(-1, 1)
#
#     for name, logit in logits.items():
#         ranking = torch.argsort(logit, descending=True)
#         preds = torch.where(ranking == ground_truth)[1]
#         preds = preds.detach().cpu().numpy()
#         metrics[f"{name}_mean_rank"] = preds.mean() + 1
#         metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
#         for k in [1, 5, 10]:
#             metrics[f"{name}_R@{k}"] = np.mean(preds < k)
#
#     return metrics


