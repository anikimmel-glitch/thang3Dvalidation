"""Per-track self-training; test inference is unreachable until selection is saved."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

from evaluation.common import ROOT, read, save, evaluate_program, record_failure
from evaluation.run import MODELS, prepare_one
from finetuning.split import validate
from finetuning.protocol import CONFIG, acceptable, selection_score, paired_report


def log(event, **fields):
    print(json.dumps(dict(event=event, time=time.time(), **fields)), flush=True)


def kind(name):
    return {"cadrille_rgb": "cadrille", "cadreasoner_native": "cadreasoner"}.get(name)


def ready(row, root, name):
    prepared = root / "prepared" / row["pilot_id"]
    return ((prepared / "preparation.json").exists()
            and (not kind(name) or (prepared / f"{kind(name)}.png").exists()))


def generation(adapter, row, root, dest, sample=False, candidate=0):
    import torch
    if (dest / "metrics.json").exists():
        return read(dest / "metrics.json")
    dest.mkdir(parents=True, exist_ok=True)
    if not ready(row, root, adapter.name):
        prepared = root / "prepared" / row["pilot_id"]
        status = "render_failed" if (prepared / "preparation.json").exists() else "reference_preparation_failed"
        record_failure(row, dest / "metrics.json", status)
    else:
        try:
            # Independent, repeatable RNG per body/candidate across GPU schedules.
            seed = int(hashlib.sha256(f'{row["sha256"]}/{candidate}/{CONFIG["seed"]}'.encode()).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            start = time.monotonic()
            code = adapter.generate(root / "prepared" / row["pilot_id"], sample=sample)
            save(dest / "generation.json", dict(seconds=time.monotonic()-start, seed=seed,
                sample=sample, max_new_tokens=1500 if adapter.name=="cadreasoner_native" else 768,
                source_text_provided=False))
            evaluate_program(row, code, dest, root / "prepared" / row["pilot_id"])
        except Exception as exc:
            record_failure(row, dest / "metrics.json", "inference_runtime_error",
                           f"{type(exc).__name__}: {exc}")
            torch.cuda.empty_cache()
    return read(dest / "metrics.json")


def evaluate(adapter, rows, root, dest):
    adapter.model.eval()
    records = {}
    for i, row in enumerate(rows):
        records[row["pilot_id"]] = generation(adapter, row, root, dest / row["pilot_id"])
        if (i + 1) % 20 == 0:
            log("evaluation_progress", model=adapter.name, phase=str(dest.name), completed=i+1, total=len(rows))
    save(dest / "summary.json", dict(expected=len(rows),
        statuses=dict(Counter(x["status"] for x in records.values())),
        selection_score=selection_score(rows, records)))
    return records


def training_inputs(adapter, prepared, program):
    import torch
    inputs = adapter.inputs(prepared)
    tokenizer = adapter.processor if adapter.name == "cad_recode_pc" else adapter.processor.tokenizer
    target = tokenizer(program, add_special_tokens=False)["input_ids"]
    limit = 1500 if adapter.name == "cadreasoner_native" else 768
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos is None or eos == tokenizer.unk_token_id:
        raise ValueError("Expected Qwen assistant end token")
    target = target + [eos]
    if not target or len(target) > limit:
        raise ValueError("Pseudo-program exceeds the benchmark generation budget")
    prompt = inputs["input_ids"]
    answer = torch.tensor([target], dtype=prompt.dtype, device=prompt.device)
    inputs["input_ids"] = torch.cat((prompt, answer), dim=1)
    mask = inputs["attention_mask"]
    inputs["attention_mask"] = torch.cat((mask, torch.ones_like(answer, dtype=mask.dtype)), dim=1)
    inputs["labels"] = torch.cat((torch.full_like(prompt, -100), answer), dim=1)
    return inputs


def pseudo_labels(adapter, rows, root, out):
    accepted = []
    receipts = []
    if any(r["split"] != "train" for r in rows):
        raise ValueError("Pseudo-label generation is restricted to training sources")
    for index, row in enumerate(rows):
        options = []
        for c in range(CONFIG["candidates_per_training_body"]):
            dest = out / "candidates" / row["pilot_id"] / str(c)
            metric = generation(adapter, row, root, dest, sample=c>0, candidate=c)
            fits = False
            if acceptable(metric):
                try:
                    inputs = training_inputs(adapter, root / "prepared" / row["pilot_id"],
                                              (dest / "prediction.py").read_text())
                    del inputs
                    fits = True
                except ValueError:
                    pass
            receipts.append(dict(pilot_id=row["pilot_id"], candidate=c, status=metric["status"],
                                 accepted_gate=acceptable(metric), fits_token_budget=fits))
            if fits:
                options.append((metric["scores"]["native_frame"]["surface_f1_at_1pct_side"], -c, dest))
        if options:
            score, _, dest = max(options)
            accepted.append(dict(row=row, program=str(dest / "prediction.py"), native_f1=score))
        save(out / "pseudo_labels.json", dict(config=CONFIG, accepted=accepted, candidates=receipts,
             completed_training_bodies=index+1, expected_training_bodies=len(rows)))
        if (index+1) % 10 == 0:
            log("pseudo_labels", model=adapter.name, completed=index+1, total=len(rows), accepted=len(accepted))
    sources = {x["row"]["source_group"] for x in accepted}
    save(out / "pseudo_label_coverage.json", dict(
        expected_bodies=len(rows), accepted_bodies=len(accepted), accepted_sources=len(sources),
        eligible_sources=len({r["source_group"] for r in rows}),
        accepted_by_face_bin=dict(Counter(x["row"]["complexity_bin"] for x in accepted)),
        expected_by_face_bin=dict(Counter(r["complexity_bin"] for r in rows)),
        note="Execution and approximate geometry checks, not ground-truth CAD history. "
             "Acceptance can favor easier shapes; rejected shapes are not silently relabeled."))
    return accepted


def optimizer_epoch(adapter, accepted, root, optimizer, epoch, steps, total_steps, out):
    import torch
    model = adapter.model
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    groups = defaultdict(list)
    for item in accepted:
        groups[item["row"]["source_group"]].append(item)
    # Equal probability of each accepted source; then uniform within that source.
    rng = random.Random(CONFIG["seed"] + epoch)
    names = sorted(groups)
    stream = [rng.choice(groups[rng.choice(names)])
              for _ in range(CONFIG.get("epoch_draws",len(accepted)))]
    losses = []
    for first in range(0, len(stream), CONFIG["accumulation"]):
        batch = stream[first:first+CONFIG["accumulation"]]
        optimizer.zero_grad(set_to_none=True)
        for item in batch:
            inputs = training_inputs(adapter, root / "prepared" / item["row"]["pilot_id"],
                                     Path(item["program"]).read_text())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**inputs, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            (loss / len(batch)).backward()
            losses.append(float(loss.detach().cpu()))
            del inputs, loss
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable):
            raise RuntimeError("Missing/nonfinite LoRA gradients")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, CONFIG["max_grad_norm"])
        warmup = max(1, math.ceil(total_steps * CONFIG["warmup_fraction"]))
        factor = min(1., (steps+1)/warmup) * max(0., (total_steps-steps)/(max(1,total_steps-warmup)))
        lr = CONFIG["learning_rate"] * min(1., factor)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        steps += 1
        record = dict(epoch=epoch, optimizer_step=steps, mean_loss=sum(losses[-len(batch):])/len(batch),
                      learning_rate=lr, gradient_norm=float(grad_norm))
        with (out / "training.jsonl").open("a") as f:
            f.write(json.dumps(record)+"\n")
        log("training_step", model=adapter.name, **record)
    model.gradient_checkpointing_disable()
    model.eval()
    return steps


def track(args, manifest):
    import torch
    from evaluation.adapters import Adapter
    from finetuning import lora
    rows = validate(manifest)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    primary = set(manifest["primary_test_ids"])
    test = [r for r in rows if r["pilot_id"] in primary]
    out = args.out / args.model
    out.mkdir(parents=True, exist_ok=True)
    # A training rerun must use a fresh output location, not stale epoch results.
    if (out / "training_started.json").exists():
        raise RuntimeError("Training output already used; resume is not implicit")
    torch.manual_seed(CONFIG["seed"])
    adapter = Adapter(args.model)
    torch.set_default_dtype(torch.float32)  # upstream constructors change this globally
    save(out / "protocol.json", dict(config=CONFIG, split_sha256=manifest["split_sha256"],
                                    model_lock=read(ROOT/"models.lock.json")))
    names = lora.install(adapter.model, CONFIG["rank"], CONFIG["alpha"])
    # Exercise real model tokenization/forward/backward before costly candidate generation.
    # This synthetic syntax canary uses a train input and is completely reset afterward.
    first = next(r for r in train if ready(r, args.out, args.model))
    initial = lora.state(adapter.model)
    adapter.model.train()
    adapter.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    inputs = training_inputs(adapter, args.out/"prepared"/first["pilot_id"],
                             "import cadquery as cq\nr=cq.Workplane('XY').box(20,30,40)")
    optimizer = torch.optim.AdamW([p for p in adapter.model.parameters() if p.requires_grad], lr=1e-4)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = adapter.model(**inputs, use_cache=False).loss
    loss.backward()
    gradients = [p.grad for p in adapter.model.parameters() if p.requires_grad]
    if not torch.isfinite(loss) or not all(g is not None and torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Training canary loss/gradient failure")
    optimizer.step()
    if not any(not torch.equal(v, initial[n]) for n,v in lora.state(adapter.model).items()):
        raise RuntimeError("Training canary did not change any adapter weights")
    lora.load(adapter.model, initial)
    adapter.model.zero_grad(set_to_none=True)
    adapter.model.gradient_checkpointing_disable()
    adapter.model.eval()
    save(out/"training_canary_passed.json", dict(subject=first["pilot_id"], loss=float(loss.detach().cpu()),
        finite_gradients=True, adapter_updated=True, all_updates_reset=True,
        synthetic_program_used_only_for_runtime_check=True))
    del inputs, loss, optimizer, gradients, initial
    torch.cuda.empty_cache()
    log("training_canary_passed", model=args.model)
    accepted = pseudo_labels(adapter, train, args.out, out)
    if (len(accepted) < CONFIG["min_accepted_bodies"] or
            len({x["row"]["source_group"] for x in accepted}) < CONFIG["min_accepted_source_groups"]):
        save(out / "insufficient_pseudo_labels.json", dict(accepted=len(accepted), config=CONFIG))
        raise RuntimeError("Insufficient quality-verified training data; gate retained unchanged")
    # Fix validation eligibility from preparation before observing model outcomes.
    val_ready = [r for r in val if ready(r, args.out, args.model)]
    save(out / "validation_coverage.json", dict(expected=len(val), ready=len(val_ready),
        excluded_preparation_ids=[r["pilot_id"] for r in val if r not in val_ready]))
    if len({r["source_group"] for r in val_ready}) < 8:
        raise RuntimeError("Too few prepared validation source groups")
    baseline = evaluate(adapter, val_ready, args.out, out / "validation_epoch_0")
    best_score = selection_score(val_ready, baseline)
    if best_score is None:
        raise RuntimeError("Incomplete baseline validation coverage; cannot select checkpoints")
    best_state = lora.state(adapter.model)
    best_epoch = 0
    save(out / "trainable_parameters.json", dict(projections=names,
        trainable=sum(p.numel() for p in adapter.model.parameters() if p.requires_grad),
        total=sum(p.numel() for p in adapter.model.parameters())))
    optimizer = torch.optim.AdamW([p for p in adapter.model.parameters() if p.requires_grad],
                                 lr=CONFIG["learning_rate"], weight_decay=0.)
    steps = 0
    total_steps = math.ceil(CONFIG.get("epoch_draws",len(accepted))/CONFIG["accumulation"]) * CONFIG["epochs"]
    history = [dict(epoch=0, validation_source_macro_f1=best_score)]
    save(out / "training_started.json", dict(time=time.time(), total_steps=total_steps))
    for epoch in range(1, CONFIG["epochs"]+1):
        steps = optimizer_epoch(adapter, accepted, args.out, optimizer, epoch, steps, total_steps, out)
        checkpoint = out / f"adapter_epoch_{epoch}.pt"
        torch.save(lora.state(adapter.model), checkpoint)
        # Preserve optimizer and RNG state for explicit, reviewed recovery; a new
        # run still never implicitly loads a partial training checkpoint.
        training_state = dict(epoch=epoch, optimizer_step=steps, total_steps=total_steps,
            optimizer=optimizer.state_dict(), adapter=lora.state(adapter.model),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(), split_sha256=manifest["split_sha256"],
            model_lock=read(ROOT/"models.lock.json"), config=CONFIG)
        temporary = out / f"training_state_epoch_{epoch}.tmp"
        torch.save(training_state, temporary)
        temporary.replace(out / f"training_state_epoch_{epoch}.pt")
        del training_state
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        records = evaluate(adapter, val_ready, args.out, out / f"validation_epoch_{epoch}")
        score = selection_score(val_ready, records)
        history.append(dict(epoch=epoch, validation_source_macro_f1=score, adapter_sha256=checkpoint_sha))
        if score is not None and score > best_score:
            best_epoch, best_score, best_state = epoch, score, lora.state(adapter.model)
        save(out / "validation_history.json", history)
    # Save and hash the selected adapter before any test inference or test metrics.
    lora.load(adapter.model, best_state)
    torch.save(best_state, out / "selected_adapter.pt")
    save(out / "selection_frozen.json", dict(epoch=best_epoch, validation_score=best_score,
        adapter_sha256=hashlib.sha256((out/"selected_adapter.pt").read_bytes()).hexdigest(),
        split_sha256=manifest["split_sha256"], time=time.time(),
        test_used_for_selection=False, baseline_selected=best_epoch==0))
    # Test subjects are first prepared after checkpoint selection, using unchanged
    # evaluation inputs/normalization. Every sibling stays protected by the manifest.
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda r: prepare_one(r,args.input,args.out,[kind(args.model)] if kind(args.model) else []), test))
    tuned = evaluate(adapter, test, args.out, out / "test_selected")
    # Reset LoRA to its zero-update state; the frozen base is the identical pinned model.
    zero = {n: torch.zeros_like(v) if n.endswith("lora_b") else v for n,v in best_state.items()}
    lora.load(adapter.model, zero)
    baseline = evaluate(adapter, test, args.out, out / "test_pretrained")
    comparison = paired_report(test, baseline, tuned)
    comparison.update(selected_epoch=best_epoch, split_sha256=manifest["split_sha256"],
                      validation_history=history, test_used_for_selection=False)
    save(out / "comparison.json", comparison)
    save(out / "completed.json", dict(time=time.time(), selected_epoch=best_epoch))
    log("track_complete", model=args.model, selected_epoch=best_epoch, paired=comparison["paired_scored"])


def run(args, manifest):
    import torch
    from huggingface_hub import snapshot_download
    rows = validate(manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / "split_manifest.json", manifest)
    save(args.out / "protocol.json", CONFIG)
    if torch.cuda.device_count() < len(MODELS):
        raise RuntimeError("Four GPU tracks required by this experiment")
    # Four models share prepared training/validation inputs; no test data here.
    optimize = [r for r in rows if r["split"] != "test"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda r: prepare_one(r,args.input,args.out,["cadrille","cadreasoner"]), optimize))
    lock = read(ROOT / "models.lock.json")
    for entry in lock.values():
        if isinstance(entry, dict) and "model_id" in entry:
            snapshot_download(entry["model_id"], revision=entry["revision"],
                allow_patterns=["*.json","*.txt","*.model","*.safetensors","*.tiktoken","*.jinja"])
    # Each track owns its prepared directory for later test rendering to avoid races.
    processes = []
    for rank, name in enumerate(MODELS):
        track_root = args.out / f"track_{name}"
        track_root.mkdir(parents=True, exist_ok=True)
        # Hard-link read-only optimization inputs on the same filesystem.
        shutil.copytree(args.out/"prepared", track_root/"prepared", copy_function=os.link, dirs_exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), HF_HUB_OFFLINE="1")
        command = [sys.executable, "-m", "finetuning.train", "--manifest", str(args.manifest),
                   "--out", str(track_root), "--input", str(args.input), "--model", name]
        if getattr(args,"epoch_draws",0):
            command += ["--epoch-draws",str(args.epoch_draws)]
        stream = (args.out/f"{name}.log").open("w")
        processes.append((name, subprocess.Popen(command, cwd=ROOT, env=env,
                          stdout=stream, stderr=subprocess.STDOUT), stream))
    while any(p.poll() is None for _,p,_ in processes):
        log("track_status", tracks={n:p.poll() for n,p,_ in processes})
        time.sleep(30)
    statuses = {}
    for name, process, stream in processes:
        stream.close()
        statuses[name] = process.returncode
        log("track_exit", model=name, code=process.returncode,
            log_tail=(args.out/f"{name}.log").read_text(errors="replace")[-3000:])
    save(args.out / "track_exit_codes.json", statuses)
    if any(statuses.values()):
        raise RuntimeError(f"Fine-tuning track failures: {statuses}; inspect retained artifacts")
    save(args.out / "completed.json", dict(time=time.time(), tracks=list(MODELS),
                                         split_sha256=manifest["split_sha256"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--input", type=Path, default=Path("/opt/ml/input/data/training"))
    p.add_argument("--out", type=Path, default=Path("/opt/ml/checkpoints/irl-cad-finetuning"))
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--epoch-draws", type=int, default=0,
                   help="Explicit study control: equal source-sampled training examples per epoch.")
    a = p.parse_args()
    if a.epoch_draws:
        if a.epoch_draws < 20:
            raise ValueError("Study epoch sample budget too small")
        CONFIG["epoch_draws"]=a.epoch_draws
    try:
        (track if a.model else run)(a, read(a.manifest))
    except Exception as exc:
        save(a.out / (f"failure_{a.model}.json" if a.model else "failure.json"),
             dict(error=str(exc), traceback=traceback.format_exc(), time=time.time()))
        raise
