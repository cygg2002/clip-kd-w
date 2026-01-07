import logging
import os
import sys
import random
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import GradScaler
from src.open_clip import ClipLoss, KDClipLoss, get_cast_dtype

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from src.open_clip import create_kd_model_and_transforms, trace_model, get_tokenizer
from src.training.data import get_data
from src.training.distributed import is_master, init_distributed_device, world_info_from_env
from src.training.logger import setup_logging
from src.training.params import parse_args
from src.training.scheduler import cosine_lr
from src.training.train import train_kd_one_epoch, evaluate

# ===== ADD: taxonomy distill loss =====
# ÄãÐèÒª°Ñ TaxoDistillLoss / TaxoDistillConfig ·ÅÔÚ src/open_clip/taxo_distill_loss.py
from src.open_clip.taxo_distill_loss import TaxoDistillLoss, TaxoDistillConfig


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


# -----------------------------
# Taxonomy helpers (JSONL)
# -----------------------------
def parse_taxo_path(taxo_path: str):
    # "class:Mammalia > order:... > family:... > genus:... > species:..."
    parts = [p.strip() for p in taxo_path.split(">")]
    out = {}
    for p in parts:
        k, v = p.split(":", 1)
        out[k.strip()] = v.strip()
    return out  # keys: class, order, family, genus, species


def build_taxo_vocab_and_maps(jsonl_paths):
    """
    Build level vocab + prompts + species->(gen/fam/ord/cls) mappings from JSONL.
    Returns:
      level_list:    dict level -> list[str] (sorted, fixed order)
      level_prompts: dict level -> list[str] aligned with level_list[level]
      sp_to_gen_id/sp_to_fam_id/sp_to_ord_id/sp_to_cls_id: torch.LongTensor [N_sp]
      gen_size/fam_size/ord_size/cls_size: torch.FloatTensor
    """
    prompts_by_level = {l: {} for l in ["species", "genus", "family", "order", "class"]}
    sp2gen, sp2fam, sp2ord, sp2cls = {}, {}, {}, {}

    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)

                taxo = parse_taxo_path(obj["taxo_path"])
                sp = taxo["species"]
                ge = taxo["genus"]
                fa = taxo["family"]
                orr = taxo["order"]
                cl = taxo["class"]

                sp2gen[sp] = ge
                sp2fam[sp] = fa
                sp2ord[sp] = orr
                sp2cls[sp] = cl

                pr = obj.get("prompts", {})
                # ÓÃÃ¿ÌõÑù±¾Ð¯´øµÄ prompt À´Îª¶ÔÓ¦Àà±ð½¨Á¢ prompt
                for lvl in prompts_by_level.keys():
                    if lvl in pr:
                        name = taxo[lvl]
                        if name not in prompts_by_level[lvl]:
                            prompts_by_level[lvl][name] = pr[lvl]

    # ¹Ì¶¨´Ê±íË³Ðò£¨±ØÐë¹Ì¶¨£©
    level_list = {}
    for lvl in ["species", "genus", "family", "order", "class"]:
        level_list[lvl] = sorted(prompts_by_level[lvl].keys())

    # prompt list Óë´Ê±í¶ÔÆë
    level_prompts = {}
    for lvl in ["species", "genus", "family", "order", "class"]:
        level_prompts[lvl] = [prompts_by_level[lvl][name] for name in level_list[lvl]]

    # ½¨Ë÷ÒýÓ³Éä
    sp_list = level_list["species"]
    gen_to_id = {g: i for i, g in enumerate(level_list["genus"])}
    fam_to_id = {g: i for i, g in enumerate(level_list["family"])}
    ord_to_id = {g: i for i, g in enumerate(level_list["order"])}
    cls_to_id = {g: i for i, g in enumerate(level_list["class"])}

    sp_to_gen_id = torch.tensor([gen_to_id[sp2gen[sp]] for sp in sp_list], dtype=torch.long)
    sp_to_fam_id = torch.tensor([fam_to_id[sp2fam[sp]] for sp in sp_list], dtype=torch.long)
    sp_to_ord_id = torch.tensor([ord_to_id[sp2ord[sp]] for sp in sp_list], dtype=torch.long)
    sp_to_cls_id = torch.tensor([cls_to_id[sp2cls[sp]] for sp in sp_list], dtype=torch.long)

    # group sizes
    def group_sizes(sp_to_group_id, n_group):
        cnt = torch.zeros(n_group, dtype=torch.float32)
        ones = torch.ones_like(sp_to_group_id, dtype=torch.float32)
        cnt.scatter_add_(0, sp_to_group_id, ones)
        return cnt.clamp_min(1.0)

    gen_size = group_sizes(sp_to_gen_id, len(level_list["genus"]))
    fam_size = group_sizes(sp_to_fam_id, len(level_list["family"]))
    ord_size = group_sizes(sp_to_ord_id, len(level_list["order"]))
    cls_size = group_sizes(sp_to_cls_id, len(level_list["class"]))

    return level_list, level_prompts, sp_to_gen_id, sp_to_fam_id, sp_to_ord_id, sp_to_cls_id, gen_size, fam_size, ord_size, cls_size


@torch.no_grad()
def encode_text_vocab(model, tokenizer, prompts, device, batch_size=256, use_amp=False):
    """
    Encode a list of prompts -> normalized text embeddings.
    Returns: [N, D]
    """
    model.eval()
    feats = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokens = tokenizer(batch_prompts).to(device)

        if use_amp:
            with torch.cuda.amp.autocast():
                f = model.encode_text(tokens)
        else:
            f = model.encode_text(tokens)

        f = F.normalize(f, dim=-1)
        feats.append(f)
    return torch.cat(feats, dim=0)


def mu_schedule(step, total_steps, warm_frac=0.3, mu_max=0.2):
    """Distance KD È¨ÖØµ÷¶È£ºÇ° warm_frac Îª 0£¬Ö®ºóÏßÐÔÉýµ½ mu_max¡£"""
    p = step / max(1, total_steps)
    if p < warm_frac:
        return 0.0
    return mu_max * (p - warm_frac) / (1.0 - warm_frac)


def freeze_text_tower(m):
    m = m.module if hasattr(m, "module") else m
    # open_clip ÀïÍ¨³£ÊÇ transformer + token_embedding + positional_embedding + ln_final + text_projection
    for name, p in m.named_parameters():
        if name.startswith("transformer") or name.startswith("token_embedding") \
           or name.startswith("positional_embedding") or name.startswith("ln_final") \
           or name.startswith("text_projection"):
            p.requires_grad = False


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    args.model = args.model.replace('/', '-')

    with open(os.path.join(os.getcwd(), 'src/open_clip/model_configs/' + args.t_model + '.json'), 'r') as f:
        args.t_embed_dim = json.load(f)['embed_dim']
    with open(os.path.join(os.getcwd(), 'src/open_clip/model_configs/' + args.model + '.json'), 'r') as f:
        args.s_embed_dim = json.load(f)['embed_dim']

    # ====== IMPORTANT ======
    # Ô¤¼ÆËã s_text_vocab ÐèÒª student text tower ²»±ä£¬
    # µÚÒ»°æÇ¿ÁÒ½¨ÒéÄã¼Ó --lock-text ¶³½á text tower
    if not args.lock_text:
        logging.warning(
            "You are NOT locking student text tower (--lock-text is False). "
            "If you precompute student text vocab embeddings once, they will become stale during training. "
            "Recommend to run with --lock-text for first stable implementation."
        )

    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"t_model_{args.t_model}",
            f"s_model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"tag_{args.tag}"
        ])

    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            print("Error. Experiment already exists. Use --name to specify a new experiment.")
            return -1

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    device = init_distributed_device(args)

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if is_master(args):
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

    if args.copy_codebase:
        copy_codebase(args)

    if args.precision == 'fp16':
        logging.warning('It is recommended to use AMP mixed-precision instead of FP16.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode. Device: {args.device}. '
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode. Device: {args.device}. '
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    random_seed(args.seed, 0)

    model, t_model, preprocess_train, preprocess_val = create_kd_model_and_transforms(
        args.model,
        args.t_model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
    )
    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        freeze_text_tower(model)
        # model.lock_text_tower(
        #     unlocked_layers=args.lock_text_unlocked_layers,
        #     freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Teacher Visual Params:")
        logging.info(f"{str(sum([i.numel() for i in t_model.visual.parameters()]) / 1e6)}M")
        logging.info("Teacher Text Params:")
        logging.info(f"{str(sum([i.numel() for i in t_model.transformer.parameters()]) / 1e6)}M")
        logging.info("Student Visual Params:")
        logging.info(f"{str(sum([i.numel() for i in model.visual.parameters()]) / 1e6)}M")
        logging.info("Student Text Params:")
        logging.info(f"{str(sum([i.numel() for i in model.transformer.parameters()]) / 1e6)}M")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    # ===== Load teacher checkpoint =====
    t_model.eval()
    for t_n, t_p in t_model.named_parameters():
        t_p.requires_grad = False

    checkpoint = torch.load(args.t_model_checkpoint, map_location='cpu')
    sd = checkpoint
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    t_model.load_state_dict(sd)
    print('Teacher model loaded successfully')

    # ===== Original KDClipLoss =====
    kd_loss = KDClipLoss(
        args=args,
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod
    ).to(device)

    # ===== ADD: Taxonomy distill loss init =====
    # TODO: ÄãÐèÒªÔÚ params.py ÀïÐÂÔö --taxo_jsonl ²ÎÊý£¬²¢ÔÚÃüÁîÐÐ´«Èë
    assert hasattr(args, "taxo_jsonl") and args.taxo_jsonl is not None, \
        "Please provide --taxo_jsonl (a JSONL file containing taxo_path and prompts)."

    logging.info(f"Building taxonomy vocabs from: {args.taxo_jsonl}")
    level_list, level_prompts, sp_to_gen, sp_to_fam, sp_to_ord, sp_to_cls, gen_size, fam_size, ord_size, cls_size = \
        build_taxo_vocab_and_maps([args.taxo_jsonl])

    logging.info(f"Vocab sizes: species={len(level_list['species'])}, genus={len(level_list['genus'])}, "
                 f"family={len(level_list['family'])}, order={len(level_list['order'])}, class={len(level_list['class'])}")

    # Tokenizers
    s_tokenizer = get_tokenizer(args.model)
    t_tokenizer = get_tokenizer(args.t_model)

    use_amp = (args.precision == "amp")

    # Student model object for encode_text (handle DDP)
    s_model_for_text = model.module if hasattr(model, "module") else model

    # Precompute vocab text embeddings (teacher + student)
    # NOTE: If you don't lock text tower, s_text_vocab will become stale during training.
    logging.info("Encoding text vocab embeddings (this may take a while)...")
    s_text_vocab = {}
    t_text_vocab = {}
    for lvl in ["species", "genus", "family", "order", "class"]:
        s_text_vocab[lvl] = encode_text_vocab(s_model_for_text, s_tokenizer, level_prompts[lvl], device=device, use_amp=use_amp)
        t_text_vocab[lvl] = encode_text_vocab(t_model,            t_tokenizer, level_prompts[lvl], device=device, use_amp=use_amp)

    cfg = TaxoDistillConfig(
        tau=4.0,
        gamma=1.0,
        mu_dist=0.2,      # max value; actual mu will be scheduled in train loop
        beta_repr=0.5,
        # lambda_level can be customized here if you want
        # lambda_level={"species":1.0,"genus":0.4,"family":0.2,"order":0.1,"class":0.05}
    )
    taxo_loss_fn = TaxoDistillLoss(cfg, s_embed_dim=args.s_embed_dim, t_embed_dim=args.t_embed_dim).to(device)
    taxo_loss_fn.set_taxonomy(
        sp_to_gen.to(device), sp_to_fam.to(device), sp_to_ord.to(device), sp_to_cls.to(device),
        gen_size.to(device),  fam_size.to(device),  ord_size.to(device),  cls_size.to(device),
    )
    taxo_loss_fn.set_text_embeddings(s_text_vocab, t_text_vocab, normalize=True)
    logging.info("TaxoDistillLoss initialized.")

    # ===== create optimizer and scaler =====
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list((model.module if hasattr(model, "module") else model).named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
                {"params": kd_loss.parameters()},
                {"params": taxo_loss_fn.parameters()},  # <--- ADD
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=(model.module if hasattr(model, "module") else model).named_parameters())
            hvd.broadcast_parameters((model.module if hasattr(model, "module") else model).state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    # ===== optionally resume =====
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            if 'epoch' in checkpoint:
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                (model.module if hasattr(model, "module") else model).load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                (model.module if hasattr(model, "module") else model).load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    # ===== datasets =====
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model))
    assert len(data), 'At least one train or eval dataset must be specified.'

    # ===== scheduler =====
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # ===== logging =====
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        wandb.init(
            project="open-clip",
            name=args.name,
            notes=args.wandb_notes,
            tags=[],
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    if 'train' not in data:
        evaluate(model, data, start_epoch, args, writer)
        return

    if args.t_eval:
        print('evaluate teacher:')
        evaluate(t_model, data, start_epoch, args, writer)

    # ===== train loop =====
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        # TODO: ÄãÐèÒªÐÞ¸Ä train_kd_one_epoch µÄÇ©Ãû£¬½ÓÊÕ taxo_loss_fn
        # ²¢ÔÚÆäÖÐÃ¿ step µ÷ÓÃ£º
        #   mu = mu_schedule(global_step, total_steps, warm_frac=0.3, mu_max=taxo_loss_fn.cfg.mu_dist)
        #   taxo_loss, parts = taxo_loss_fn(...)
        #   total_loss += taxo_loss
        train_kd_one_epoch(model, t_model, data, epoch, kd_loss, taxo_loss_fn, optimizer, scaler, scheduler, args, writer)

        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, writer)

        # ===== save checkpoints =====
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": (model.module if hasattr(model, "module") else model).state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
                )

    logging.info(f'The files are saved at {args.logs}')
    if args.wandb and is_master(args):
        wandb.finish()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
