#!/usr/bin/env python
"""Train the CTC reader on the frozen audio encoder of EmbeddingGemma 2. One device only.

  python scripts/train.py --data_dir data/prepared --out_dir runs/base --reader base
  python scripts/train.py --data_dir data/prepared --eval_only --resume runs/base/best.pt --eval_split test

The last line of every run is "@@" and one JSON object with the final metrics.
"""

import argparse
import json
import math
import os
import sys
import threading
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eg2tamil.data import EvalSet, TrainStream, list_shards  # noqa: E402
from eg2tamil.model import READERS, GemmaCTC, ModelConfig, greedy_decode, load_encoder  # noqa: E402
from eg2tamil.scoring import ErrorCounts, score_batch  # noqa: E402
from eg2tamil.units import Units  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/prepared")
    ap.add_argument("--units", default=None, help="units JSON. Default: <data_dir>/units_gemma.json. A checkpoint has its own copy")
    ap.add_argument("--out_dir", default="runs/tamil")
    ap.add_argument("--train_sources", nargs="*", default=None, help="default: all sources in <data_dir>/train")
    ap.add_argument("--source_weights", default="{}", help='JSON, for example \'{"prajwal_cv": 2, "mile": 0.5}\'')
    ap.add_argument("--eval_split", default="valid")
    ap.add_argument("--max_eval_utts", type=int, default=None, help="per source. Default 1000, or no cap with --eval_only")
    ap.add_argument("--eval_only", action="store_true", help="evaluate --resume on --eval_split and exit")
    # model. A flag for the reader size wins over the --reader preset.
    ap.add_argument("--encoder_id", default="google/embeddinggemma-2")
    ap.add_argument("--reader", choices=list(READERS), default="base")
    ap.add_argument("--d_model", type=int)
    ap.add_argument("--n_layers", type=int)
    ap.add_argument("--n_heads", type=int)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--encoder_layers", type=int, default=12, help="run only the first K encoder layers")
    ap.add_argument("--no_layer_mix", action="store_true", help="use the last encoder layer only")
    ap.add_argument("--unfreeze_top_k", type=int, default=0, help="fine-tune the top K encoder layers")
    ap.add_argument("--head_init", choices=["text", "random"], default="text")
    ap.add_argument("--no_specaug", action="store_true")
    # optimization
    ap.add_argument("--max_steps", type=int, default=20000, help="optimizer steps")
    ap.add_argument("--max_minutes", type=float, default=None, help="wall-clock budget. Then: eval, save, exit")
    ap.add_argument("--max_batch_seconds", type=float, default=600.0, help="padded audio seconds per micro-batch")
    ap.add_argument("--grad_accum", type=int, default=1, help="micro-batches per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=5e-5, help="lr of the unfrozen encoder layers")
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--min_lr_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--stall_minutes", type=float, default=10.0, help="exit with an error if no step ends in this time")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--resume", default=None, help="checkpoint path, for example runs/base/last.pt")
    return ap.parse_args()


_beat = [time.time()]  # time of the last sign of life: an optimizer step or an eval batch


def fail(msg):
    """Print the error as the final line and exit with a non-zero code. `os._exit` also works from the watchdog thread."""
    print(f"ERROR: {msg}\n@@" + json.dumps({"error": msg}), flush=True)
    os._exit(3)


def watch(stall_minutes):
    def loop():
        while True:
            time.sleep(15)
            if time.time() - _beat[0] > stall_minutes * 60:
                fail(f"stall: no optimizer step for {stall_minutes} min")

    threading.Thread(target=loop, daemon=True).start()


def lr_factor(step, warmup, total, min_ratio):
    if step < warmup:
        return (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, units, eval_sets, device, amp, max_batch_seconds):
    """Greedy-decode every set. Return {source: summary + hours, "avg_wer": mean WER over the sources}."""
    model.eval()
    results = {}
    for name, es in eval_sets.items():
        counts, example = ErrorCounts(), None
        for batch in es.batches(max_batch_seconds):
            _beat[0] = time.time()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits, out_lens = model(batch["wav"].to(device), batch["wav_lens"].to(device))
            hyps = [units.decode(u) for u in greedy_decode(logits, out_lens)]
            counts += score_batch(batch["texts"], hyps)
            example = (batch["texts"][-1], hyps[-1])
        results[name] = {**counts.summary(), "hours": es.hours}
        r = results[name]
        print(f"  [eval] {name:16s} WER {r['wer']:7.2%}  CER {r['cer']:7.2%}  {r['utts']} utts  {es.hours:.2f} h", flush=True)
        if example:
            print(f"         ref: {example[0]}\n         hyp: {example[1]}", flush=True)
    model.train()
    if results:
        results["avg_wer"] = sum(r["wer"] for r in results.values()) / len(results)
    return results


def main():
    # ponytail: single GPU only. Add DDP when we get more than one GPU.
    args = parse_args()
    t_start = time.time()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        assert torch.cuda.is_bf16_supported(), "This GPU has no bf16 (a T4, for example). Use an L4 or an A100."
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    # The checkpoint holds the full units JSON. A copy goes to out_dir, because the data workers read a file.
    if ckpt:
        units_json = ckpt["units"]
    else:
        with open(args.units or os.path.join(args.data_dir, "units_gemma.json")) as f:
            units_json = json.load(f)
    units_path = os.path.join(args.out_dir, "units.json")
    with open(units_path, "w") as f:
        json.dump(units_json, f)
    units = Units.load(units_path)
    if args.eval_only:
        assert ckpt, "--eval_only needs --resume"
    if ckpt:  # the checkpoint holds the model config, so reader flags do not matter here
        model = GemmaCTC.from_checkpoint(ckpt)
    else:
        cfg = ModelConfig(
            encoder_id=args.encoder_id, vocab_size=len(units), dropout=args.dropout, layer_mix=not args.no_layer_mix,
            encoder_layers=args.encoder_layers, unfreeze_top_k=args.unfreeze_top_k, head_init=args.head_init,
            **{k: getattr(args, k) or v for k, v in READERS[args.reader].items()},
        )
        encoder, unit_emb = load_encoder(args.encoder_id, units.gemma_ids)
        model = GemmaCTC(cfg, encoder, unit_emb)
    model.to(device)
    # ponytail: no torch.compile. The chunked encoder attention recompiles for each length mod 12 and hits the
    # recompile limit (17.8 s/step against 1.33 s/step eager on an L4). Upgrade: pad audio to a multiple of 0.48 s.
    cfg = model.cfg
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"units {len(units)}  device {device}  trainable {n_train / 1e6:.1f}M  reader d={cfg.d_model} "
          f"layers={cfg.n_layers} heads={cfg.n_heads}  encoder_layers={cfg.encoder_layers}  unfreeze_top_k={cfg.unfreeze_top_k}", flush=True)

    max_utts = args.max_eval_utts if args.max_eval_utts is not None else (None if args.eval_only else 1000)
    eval_sets = {n: EvalSet(f, max_utts, seed=args.seed) for n, f in list_shards(args.data_dir, args.eval_split).items()}
    print(f"{args.eval_split} sets: " + ", ".join(f"{n} ({e.hours:.2f} h)" for n, e in eval_sets.items()), flush=True)
    amp = cuda  # bf16 autocast on the GPU. The CPU runs fp32 (smoke tests only).

    if args.eval_only:
        res = evaluate(model, units, eval_sets, device, amp, args.max_batch_seconds)
        print(f"\n{'source':16s} {'utts':>6s} {'hours':>7s} {'WER (lower is better)':>22s} {'CER (lower is better)':>22s}")
        for n, r in res.items():
            if n != "avg_wer":
                print(f"{n:16s} {r['utts']:6d} {r['hours']:7.2f} {r['wer']:22.2%} {r['cer']:22.2%}")
        print(f"{'average':16s} {'':6s} {'':7s} {res.get('avg_wer', float('nan')):22.2%}")
        print("@@" + json.dumps({"mode": "eval", "split": args.eval_split, "step": ckpt["step"], "eval": res}))
        return

    # ---- optimizer
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    enc = [p for n, p in named if n.startswith("encoder.")]
    rest = [(n, p) for n, p in named if not n.startswith("encoder.")]
    groups = [
        {"params": [p for _, p in rest if p.ndim >= 2], "weight_decay": args.weight_decay, "lr": args.lr},
        {"params": [p for _, p in rest if p.ndim < 2], "weight_decay": 0.0, "lr": args.lr},
    ]
    if enc:
        groups.append({"params": enc, "weight_decay": args.weight_decay, "lr": args.encoder_lr})
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), fused=cuda)

    def virtual_step(s):
        """With --max_minutes, the schedule follows the steps or the clock, whichever is further on.
        So the cosine ends at the time budget, even when the speed is not known before the run."""
        # ponytail: the clock restarts on --resume, so a resumed budget run starts its time share again.
        if args.max_minutes is None:
            return s
        return max(s, int(args.max_steps * (time.time() - t_start) / (args.max_minutes * 60)))

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(virtual_step(s), args.warmup_steps, args.max_steps, args.min_lr_ratio))
    step, best = 0, float("inf")
    if ckpt:
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        step, best = ckpt["step"], ckpt["best"]
        print(f"resumed from {args.resume} at step {step}", flush=True)

    # ---- data
    files = [f for fs in list_shards(args.data_dir, "train", args.train_sources).values() for f in fs]
    assert files, "no training shards found"
    stream = TrainStream(files, units_path, seed=args.seed + step, max_batch_seconds=args.max_batch_seconds,
                         source_weights=json.loads(args.source_weights))
    nw = args.num_workers
    loader = DataLoader(stream, batch_size=None, num_workers=nw, pin_memory=cuda,
                        prefetch_factor=4 if nw else None, persistent_workers=nw > 0)

    def save(name, **extra):  # atomic: write a temp file, then rename it
        tmp = os.path.join(args.out_dir, name + ".tmp")
        torch.save({"model": model.trainable_state_dict(), "config": model.config_dict(), "units": units_json,
                    "step": step, "best": best, "args": vars(args), **extra}, tmp)
        os.replace(tmp, os.path.join(args.out_dir, name))

    metrics = open(os.path.join(args.out_dir, "metrics.jsonl"), "a")

    def do_eval():
        nonlocal best
        print(f"[eval @ step {step}]", flush=True)
        res = evaluate(model, units, eval_sets, device, amp, args.max_batch_seconds)
        if res:
            metrics.write(json.dumps({"step": step, "eval": res}) + "\n")
            metrics.flush()
            if res["avg_wer"] < best:
                best = res["avg_wer"]
                save("best.pt", eval=res)
                print(f"  new best average WER {best:.2%} -> best.pt", flush=True)
        return res

    # ---- train
    model.train()
    data = iter(loader)
    res, n_logs = {}, 0
    t0, audio, loss_sum_log, n_micro, skipped = time.time(), 0.0, 0.0, 0, 0
    stop, nan_steps = False, 0
    _beat[0] = time.time()
    watch(args.stall_minutes)
    while step < args.max_steps and not stop:
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            b = next(data)
            wav, lens = b["wav"].to(device, non_blocking=True), b["wav_lens"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                logits, out_lens = model(wav, lens, specaug=not args.no_specaug)
            tl = b["target_lens"].to(device)
            logp = F.log_softmax(logits, -1).transpose(0, 1)  # (T, B, V), fp32
            loss = F.ctc_loss(logp, b["targets"].to(device), out_lens, tl, blank=0, reduction="sum",
                              zero_infinity=True) / tl.sum().clamp(min=1)
            (loss / args.grad_accum).backward()
            audio += lens.sum().item() / 16000
            step_loss += loss.item()
            n_micro += 1
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        if torch.isfinite(gnorm):
            opt.step()
        else:
            skipped += 1
        sched.step()
        step += 1
        _beat[0] = time.time()
        loss_sum_log += step_loss
        nan_steps = nan_steps + 1 if not math.isfinite(step_loss) else 0
        if nan_steps >= 50:
            fail(f"the loss was not finite for 50 steps in a row (step {step})")

        if step % args.log_every == 0:
            dt = time.time() - t0
            rec = {"step": step, "loss": loss_sum_log / n_micro, "lr": sched.get_last_lr()[0], "grad_norm": gnorm.item(),
                   "s_per_step": dt / args.log_every, "x_realtime": audio / dt, "skipped": skipped,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0}
            n_logs += 1
            print(f"step {step:6d}  loss {rec['loss']:.3f}  lr {rec['lr']:.2e}  gnorm {rec['grad_norm']:.2f}  "
                  f"{rec['s_per_step']:.2f} s/step  {rec['x_realtime']:.0f}x realtime  mem {rec['peak_mem_gb']:.1f} GB", flush=True)
            if model.layer_logits is not None and n_logs % 10 == 0:
                rec["layer_mix"] = [round(x, 3) for x in torch.softmax(model.layer_logits.float(), 0).tolist()]
                print(f"  layer mix (subsampler, then layers 1..K): {rec['layer_mix']}", flush=True)
            metrics.write(json.dumps(rec) + "\n")
            metrics.flush()
            t0, audio, loss_sum_log, n_micro = time.time(), 0.0, 0.0, 0

        stop = args.max_minutes is not None and time.time() - t_start > args.max_minutes * 60
        if stop:
            print(f"time budget of {args.max_minutes} min used up at step {step}", flush=True)
        elif step % args.eval_every == 0 and step < args.max_steps:
            res = do_eval()
        if step % args.save_every == 0:
            save("last.pt", optimizer=opt.state_dict(), scheduler=sched.state_dict())

    res = do_eval()  # final eval, then the final checkpoint
    save("last.pt", optimizer=opt.state_dict(), scheduler=sched.state_dict())
    print("@@" + json.dumps({"mode": "train", "step": step, "best_avg_wer": best, "skipped_steps": skipped,
                             "minutes": round((time.time() - t_start) / 60, 2), "eval": res}))


if __name__ == "__main__":
    main()
