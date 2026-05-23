import os
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from omegaconf import OmegaConf
import wandb

from src.model.fastspeech2 import FastSpeech2
from src.loss import FastSpeech2Loss
from src.data.dataset import LibriTTSDataset
from src.data.collate import collate_fn


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def noam_lr(step, d_model, warmup):
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(cfg):
    datasets = []
    for subset in cfg.data.subsets:
        manifest = os.path.join(cfg.paths.preprocessed, f"{subset}_manifest.txt")
        datasets.append(LibriTTSDataset(manifest, os.path.join(cfg.paths.preprocessed, subset)))
    val_manifest = os.path.join(cfg.paths.preprocessed, f"{cfg.data.val_subset}_manifest.txt")
    val_ds = LibriTTSDataset(val_manifest, os.path.join(cfg.paths.preprocessed, cfg.data.val_subset), split="val")
    return ConcatDataset(datasets), val_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fastspeech2.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    local_rank = setup_ddp()
    is_main = local_rank == 0
    seed_everything(cfg.project.seed + local_rank)

    if is_main:
        os.makedirs(cfg.paths.checkpoints, exist_ok=True)
        os.makedirs(cfg.paths.logs, exist_ok=True)
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags,
            dir=cfg.paths.wandb_dir,
            resume="allow",
        )

    device = torch.device(f"cuda:{local_rank}")
    model = FastSpeech2(cfg).to(device)

    if cfg.train.compile:
        model = torch.compile(model)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    criterion = FastSpeech2Loss(cfg).to(device)

    raw = model.module if hasattr(model, "module") else model
    raw.variance_adaptor.set_pitch_bins(cfg.audio.pitch_min, cfg.audio.pitch_max, cfg.audio.pitch_log_scale)
    raw.variance_adaptor.set_energy_bins(cfg.audio.energy_min, cfg.audio.energy_max)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: noam_lr(step, cfg.scheduler.d_model, cfg.scheduler.warmup_steps)
    )
    scaler = GradScaler(enabled=cfg.train.mixed_precision)

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]

    train_ds, val_ds = build_datasets(cfg)
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, sampler=train_sampler,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    model.train()
    opt.zero_grad()
    accum = 0

    while step < cfg.train.total_steps:
        train_sampler.set_epoch(step // len(train_loader))
        for batch in train_loader:
            if step >= cfg.train.total_steps:
                break

            phonemes = batch["phonemes"].to(device)
            mel_target = batch["mel"].to(device)
            pitch = batch["pitch"].to(device)
            energy = batch["energy"].to(device)
            duration = batch["duration"].to(device)
            src_lens = batch["src_lens"].to(device)
            mel_lens = batch["mel_lens"].to(device)

            with autocast(enabled=cfg.train.mixed_precision):
                mel_out, mel_post, log_dur, pitch_pred, energy_pred, \
                    src_mask, mel_mask, _, _ = model(
                        phonemes, src_lens, mel_lens, mel_target.size(1),
                        pitch, energy, duration
                    )
                loss, ml, pl_post, dl, ptl, el = criterion(
                    (mel_out, mel_post, log_dur, pitch_pred, energy_pred),
                    (mel_target, duration, pitch, energy),
                    src_mask, mel_mask,
                )
                loss = loss / cfg.train.grad_accum_steps

            scaler.scale(loss).backward()
            accum += 1

            if accum == cfg.train.grad_accum_steps:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_thresh)
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                opt.zero_grad()
                accum = 0
                step += 1

                if is_main and step % cfg.train.log_every == 0:
                    wandb.log({
                        "train/loss": loss.item() * cfg.train.grad_accum_steps,
                        "train/mel_loss": ml.item(),
                        "train/postnet_loss": pl_post.item(),
                        "train/duration_loss": dl.item(),
                        "train/pitch_loss": ptl.item(),
                        "train/energy_loss": el.item(),
                        "lr": scheduler.get_last_lr()[0],
                        "step": step,
                    })

                if is_main and step % cfg.train.eval_every == 0:
                    run_eval(model, val_loader, criterion, device, cfg, step)

                if is_main and step % cfg.train.save_every == 0:
                    ckpt_path = os.path.join(cfg.paths.checkpoints, f"step_{step:06d}.pt")
                    torch.save({
                        "step": step,
                        "model": raw.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "cfg": OmegaConf.to_container(cfg),
                    }, ckpt_path)


@torch.no_grad()
def run_eval(model, loader, criterion, device, cfg, step):
    model.eval()
    losses = []
    for batch in loader:
        phonemes = batch["phonemes"].to(device)
        mel_target = batch["mel"].to(device)
        pitch = batch["pitch"].to(device)
        energy = batch["energy"].to(device)
        duration = batch["duration"].to(device)
        src_lens = batch["src_lens"].to(device)
        mel_lens = batch["mel_lens"].to(device)

        mel_out, mel_post, log_dur, pitch_pred, energy_pred, \
            src_mask, mel_mask, _, _ = model(
                phonemes, src_lens, mel_lens, mel_target.size(1),
                pitch, energy, duration
            )
        loss, *_ = criterion(
            (mel_out, mel_post, log_dur, pitch_pred, energy_pred),
            (mel_target, duration, pitch, energy),
            src_mask, mel_mask,
        )
        losses.append(loss.item())

    wandb.log({"val/loss": np.mean(losses), "step": step})
    model.train()


if __name__ == "__main__":
    main()