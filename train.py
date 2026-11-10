from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata as md
import json
import math
import random
import time
import warnings

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from safetensors.torch import load_file
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from momentfm import MOMENTPipeline
from momentfm.data.pads_classification_dataset import PADS_dataset
from reconstruction.model import (ReconstructionModel, QuestionnaireFusionHead,
                                  reconstruction_loss, masked_reconstruction_loss, questionnaire_mask, load_cache, balanced_class_weights,
                                  weighted_classification_loss, movement_amplitude)


ROOT = Path(__file__).resolve().parent

CONFIG = {
    "base_path": str(ROOT / "dataset" / "preprocessed"),
    "local_model_dir": str(ROOT),
    "statement_json": str(ROOT / "questionnaire_statements.json"),
    "text_cache": str(ROOT / "outputs" / "reconstruction_text" / "reconstruction_features.npz"),
    "run_dir": str(ROOT / "outputs" / "questionnaire_amplitude_masked_mse"),
    "text_model": str(ROOT / "pretrained" / "biolord"),
    "reencode_text": False,
    "save_model": True,

    "text_batch_size": 32,
    "text_max_length": 128,

    "epochs": 15,
    "seq_len": 976,
    "micro_batch_size": 16,
    "effective_batch_size": 32,
    "num_workers": 0,
    "seed": 42,
    "val_ratio": 0.10,
    "test_ratio": 0.10,

    "reconstruction_weight": 0.01,
    "questionnaire_mask_ratio": 0.3,
    "use_amplitude_features": True,
    "fusion_time_dim": 128,
    "fusion_question_dim": 32,
    "fusion_hidden_dim": 64,
    "decoder_hidden_dim": 512,
    "decoder_lr": 1e-3,
    "lora_r": 64,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q", "v"],
    "lora_lr": 5e-5,
    "head_lr": 1e-3,

    "weight_decay": 0.01,
    "adam_betas": (0.9, 0.999),
    "adam_eps": 1e-8,
    "min_lr": 0.0,
    "label_smoothing": 0.0,
    "head_channel_dim": 16,
    "head_dropout": 0.3,
    "patch_dropout": 0.1,
    "backbone_dropout": 0.1,
    "decoder_init_std": 1e-3,
    "decoder_norm_eps": 1e-5,
    "gradient_clip": 1.0,
    "torch_num_threads": 4,
    "progress_update_batches": 10,
    "device": "auto",
    "precision": "auto",
    "gradient_checkpointing": True,
}



def emit(event, **values):
    """Console contains epoch/test metrics only; full details remain in JSON logs."""
    if event == "oom_retry":
        print(
            f"[CUDA OOM] GPU 显存不足：micro_batch_size 从 "
            f"{values['previous_micro_batch_size']} 降为 {values['next_micro_batch_size']}；"
            "将从预训练初始权重重新开始训练，有效 batch size 保持不变。",
            flush=True)
    elif event == "epoch_complete":
        val = values["validation"]
        print(
            f"Epoch {values['epoch']} | Train loss={values['train_loss']:.6f} "
            f"(CE={values['train_classification_loss']:.6f}, "
            f"MSE={values['train_reconstruction_loss']:.6f}) | "
            f"Train accuracy={values['train_accuracy']:.2%} | "
            f"Polarity accuracy={values['train_polarity_accuracy']:.2%} | "
            f"Val loss (CE)={val['classification_loss']:.6f} | "
            f"Val accuracy={val['accuracy']:.2%} | "
            f"Val macro-F1={val['macro_f1']:.4f} | "
            f"Val balanced accuracy={val['balanced_accuracy']:.2%}", flush=True)
        if "masked_mse_paired" in val:
            print(f"Val masked MSE | Paired={val['masked_mse_paired']:.6f} | "
                  f"Shuffled={val['masked_mse_shuffled']:.6f} | "
                  f"Visible-only ablation={val['masked_mse_visible_only']:.6f} | "
                  f"Train prior={val['masked_mse_train_prior']:.6f}", flush=True)
    elif event == "training_complete":
        metrics = values["test"]
        macro = metrics["classification_report"]["macro avg"]
        print(
            f"Test loss (CE)={metrics['classification_loss']:.6f} | "
            f"Accuracy={metrics['accuracy']:.2%} | Macro-F1={metrics['macro_f1']:.4f} | "
            f"Balanced accuracy={metrics['balanced_accuracy']:.2%} | "
            f"Macro precision={macro['precision']:.4f} | Macro recall={macro['recall']:.4f}", flush=True)
        print("Confusion matrix (rows=true, columns=predicted): "
              + json.dumps(metrics["confusion_matrix"]), flush=True)


def resolve_device(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def resolve_dtype(device, precision):
    if device.type != "cuda":
        if precision not in ("auto", "float32"):
            raise ValueError("CPU training supports precision=auto or float32")
        return torch.float32
    if precision == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[precision]


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, label = self.dataset[index]
        return x, label, index


def amp_context(device, dtype):
    return torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" and dtype != torch.float32 else nullcontext()


def make_model(args):
    from peft import LoraConfig, get_peft_model
    from momentfm.common import TASKS

    use_amplitude = getattr(args, "use_amplitude_features", False)
    if use_amplitude and not all(hasattr(args, key) for key in ("amplitude_mean", "amplitude_std")):
        raise ValueError("Amplitude normalization must be fitted on training subjects before make_model")
    directory = Path(args.local_model_dir)
    config = json.loads((directory / "config.json").read_text())
    config["task_name"] = TASKS.RECONSTRUCTION
    config["t5_config"]["dropout_rate"] = args.backbone_dropout
    backbone = MOMENTPipeline(config, model_kwargs={
        "task_name": "classification", "n_channels": 132, "num_class": 3,
        "seq_len": args.seq_len, "freeze_encoder": True, "freeze_embedder": True,
        "freeze_head": False, "reduction": "concat", "enable_gradient_checkpointing": args.gradient_checkpointing,
        "head_dropout": args.head_dropout, "patch_dropout": args.patch_dropout,
    })
    weights = load_file(str(directory / "model.safetensors"), device="cpu")
    backbone.load_state_dict(weights, strict=True)
    del weights
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Only best head is pre-trained.*", category=UserWarning)
        backbone.init()
    backbone = get_peft_model(backbone, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        modules_to_save=["head"], bias="none",
    ))
    if args.gradient_checkpointing:
        backbone.get_base_model().encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = ReconstructionModel(backbone, hidden_dim=args.decoder_hidden_dim,
                                init_std=args.decoder_init_std, norm_eps=args.decoder_norm_eps)
    model.fusion_head = QuestionnaireFusionHead(
        model.d_model, model.n_channels, channel_dim=args.head_channel_dim,
        time_dim=args.fusion_time_dim, question_dim=args.fusion_question_dim,
        hidden_dim=args.fusion_hidden_dim, dropout=args.head_dropout,
        amplitude_mean=args.amplitude_mean if use_amplitude else None,
        amplitude_std=args.amplitude_std if use_amplitude else None)
    model.moment.get_base_model().head.requires_grad_(False)
    model.masked_context = torch.nn.Sequential(
        torch.nn.Linear(60, model.d_model), torch.nn.GELU(),
        torch.nn.LayerNorm(model.d_model))
    if args.reconstruction_weight == 0:
        for module in (model.time_norm, model.question_norm, model.decoder, model.masked_context):
            module.requires_grad_(False)
    return model


@torch.inference_mode()
def evaluate(model, loader, device, dtype, label_smoothing=0.0, class_weights=None, answers=None,
             conditions=None, masked_evaluation=None):
    """Evaluate with subject-aligned real answers for fusion checkpoints."""
    model.eval()
    labels, predictions, probabilities = [], [], []
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    pooled_features = []
    masked_features, masked_answers = [], []
    for batch in loader:
        x, y = batch[:2]
        questionnaire = answers[batch[2].to(answers.device)] if answers is not None else None
        x, y = x.to(device, non_blocking=True).float(), y.to(device, non_blocking=True)
        with amp_context(device, dtype):
            logits, features = model(x, questionnaire)
        if masked_evaluation is not None:
            masked_features.append(features.float())
            masked_answers.append(questionnaire)
        pooled_features.append(features.float().reshape(-1, model.d_model, model.n_channels).mean(-1))
        loss_sum += weighted_classification_loss(logits, y, class_weights,
                                                 label_smoothing, reduction="sum").detach()
        count += len(y)
        labels.append(y)
        predictions.append(logits.argmax(dim=-1))
        probabilities.append(logits.float().softmax(dim=-1))
    pooled = torch.cat(pooled_features)
    unit = F.normalize(pooled, dim=-1)
    cosine = unit @ unit.T
    off_diagonal = ~torch.eye(count, dtype=torch.bool, device=device)
    statistics = torch.stack((loss_sum / count, pooled.var(0, unbiased=False).mean(),
                              cosine[off_diagonal].mean() if count > 1 else pooled.new_zeros(()))).cpu().tolist()
    # Keep batch outputs on device; transfer once at the end for sklearn/JSON.
    outputs = torch.cat((torch.cat(labels)[:, None], torch.cat(predictions)[:, None],
                         torch.cat(probabilities)), dim=1).cpu()
    labels = outputs[:, 0].long().tolist()
    predictions = outputs[:, 1].long().tolist()
    probabilities = outputs[:, 2:].tolist()
    auxiliary_metrics = {}
    if masked_evaluation is not None:
        features = torch.cat(masked_features)
        actual = torch.cat(masked_answers)
        mask = masked_evaluation["mask"]
        prototypes = masked_evaluation["prototypes"]
        target = prototypes[torch.arange(30, device=device)[None], actual.long()]
        reconstruction_inputs = (
            ("paired", features, True), ("shuffled", features.roll(1, 0), True),
            ("visible_only", features, False))
        for name, feature, motion in reconstruction_inputs:
            pred = model.reconstruct(feature, conditions, actual, mask, use_motion=motion)
            auxiliary_metrics["masked_mse_" + name] = masked_reconstruction_loss(pred, target, mask).item()
        prior = masked_evaluation["prior"].expand_as(target)
        auxiliary_metrics["masked_mse_train_prior"] = masked_reconstruction_loss(prior, target, mask).item()
        auxiliary_metrics["masked_question_count"] = int(mask.sum())
    return {
        **auxiliary_metrics,
        "pooled_feature_variance": statistics[1],
        "pooled_feature_mean_cosine": statistics[2] if count > 1 else None,
        "classification_loss": statistics[0],
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, labels=[0, 1, 2], average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(labels, predictions, labels=[0, 1, 2],
                                                         output_dict=True, zero_division=0),
        "subjects": count,
        "predicted_counts": np.bincount(predictions, minlength=3).tolist(),
    }, {"true_labels": labels, "predictions": predictions, "probabilities": probabilities}


def trainable_snapshot(model):
    """Detached CPU copy of the trainable parameters only; frozen MOMENT stays on disk."""
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().to("cpu", copy=True)
            for name, value in model.state_dict().items() if name in names}


def write_checkpoint(path, args, state, epoch, validation):
    """Atomic write; the payload format matches what predict.load_model expects."""
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save({"configuration": vars(args), "trainable_state": state, "epoch": epoch,
                "validation": validation, "selection": "best validation macro-F1"}, temp)
    temp.replace(path)


def fit_amplitude_statistics(dataset, batch_size):
    """Fit on the training dataset in its configured sequence length."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        generator=torch.Generator().manual_seed(0))
    amplitudes = torch.cat([movement_amplitude(x) for x, _ in loader]).double()
    return (amplitudes.mean(0).float().tolist(),
            amplitudes.std(0, correction=0).clamp_min(1e-6).float().tolist())


def run(args):
    started = time.time()
    seed_all(args.seed)
    torch.set_num_threads(args.torch_num_threads)
    device = resolve_device(args.device)
    args.device = str(device)
    dtype = resolve_dtype(device, args.precision)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets = {phase: PADS_dataset(args, phase) for phase in ["train", "val", "test"]}
    class_weights = balanced_class_weights(datasets["train"].labels).to(device)
    args.class_weights = class_weights.cpu().tolist()
    args.training_class_counts = np.bincount(datasets["train"].labels, minlength=3).tolist()
    splits = {phase: data.subject_ids for phase, data in datasets.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if set(splits[a]) & set(splits[b]):
            raise RuntimeError("Overlapping subject splits")
    write_json(run_dir / "subject_splits.json", splits)
    if getattr(args, "use_amplitude_features", False):
        args.amplitude_mean, args.amplitude_std = fit_amplitude_statistics(
            datasets["train"], args.micro_batch_size)
    answers, conditions, prototypes = load_cache(args.text_cache, splits["train"])
    answers, conditions, prototypes = [t.to(device) for t in (answers, conditions, prototypes)]
    targets_bank = prototypes[torch.arange(30, device=device)[None], answers]
    prior = targets_bank.mean(0, keepdim=True)
    prior_mse = reconstruction_loss(prior.expand_as(targets_bank), targets_bank).item()
    auxiliary_enabled = args.reconstruction_weight != 0
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(IndexedDataset(datasets["train"]), batch_size=args.micro_batch_size,
                              shuffle=True, generator=generator, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", drop_last=False)
    eval_loaders = {phase: DataLoader(IndexedDataset(datasets[phase]), batch_size=args.micro_batch_size,
                                     shuffle=False, num_workers=args.num_workers,
                                     pin_memory=device.type == "cuda") for phase in ["val", "test"]}
    phase_answers = {phase: load_cache(args.text_cache, splits[phase])[0].to(device)
                     for phase in ("val", "test")}
    mask_generator = torch.Generator().manual_seed(args.seed + 103)
    masked_evaluations = {}
    for index, phase in enumerate(("val", "test")):
        mask = questionnaire_mask(len(splits[phase]), args.questionnaire_mask_ratio,
                                  torch.Generator().manual_seed(args.seed + 211 + index), device)
        masked_evaluations[phase] = {"mask": mask, "prototypes": prototypes, "prior": prior}
        write_json(run_dir / (phase + "_questionnaire_masks.json"),
                   {"subject_ids": splits[phase], "mask": mask.cpu().tolist()})
    model = make_model(args).to(device)
    lora_parameters = [p for name, p in model.named_parameters() if "lora_" in name and p.requires_grad]
    classifier_parameters = [p for name, p in model.named_parameters()
                             if p.requires_grad and (name.startswith("moment.") or name.startswith("fusion_head.")) and "lora_" not in name]
    decoder_parameters = [p for name, p in model.named_parameters()
                          if p.requires_grad and not name.startswith(("moment.", "fusion_head."))]
    optimizer = torch.optim.AdamW([
        {"name": "lora", "params": lora_parameters, "lr": args.lora_lr},
        {"name": "classifier", "params": classifier_parameters, "lr": args.head_lr},
        {"name": "decoder", "params": decoder_parameters, "lr": args.decoder_lr},
    ], weight_decay=args.weight_decay, betas=tuple(args.adam_betas), eps=args.adam_eps)
    accumulation = args.effective_batch_size // args.micro_batch_size
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * updates_per_epoch, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == torch.float16)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    configuration = {
        **vars(args), "optimizer": "AdamW",
        "classification_loss": "weighted_cross_entropy",
        "class_weight_formula": "N_train / (3 * class_count_train)",
        "classification_reduction": "sum(weighted per-subject losses) / subject_count",
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                 "dropout": args.lora_dropout, "target_modules": args.lora_target_modules},
        "save_model": args.save_model, "early_stopping": False,
        "checkpoint_selection": "best validation macro-F1",
        "auxiliary_enabled": auxiliary_enabled,
        "architecture": "joint_masked_reconstruction",
        "classification_input": "movement_and_questionnaire",
        "questionnaire_input": "30 binary answers in cache question order",
        "reconstruction_input": "movement_and_visible_answers",
        "reconstruction_backprop": auxiliary_enabled,
        "text_encoder_loaded_in_training": False,
        "reconstruction_train_mean_baseline": prior_mse,
        "time_reconstruction_pooling": "mean of patches and channels; classifier retains concat",
        "precision": str(dtype) if device.type == "cuda" else "float32",
        "reconstruction_precision": "float32 decoder and MSE outside autocast",
        "trainable_parameters": sum(p.numel() for p in trainable_parameters),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "split_sizes": {phase: len(data) for phase, data in datasets.items()},
        "text_cache_sha256": hashlib.sha256(Path(args.text_cache).read_bytes()).hexdigest(),
        "packages": {name: md.version(name) for name in ["torch", "transformers", "peft", "numpy", "scikit-learn"]},
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    write_json(run_dir / "configuration.json", configuration)
    print(f"Training started | device={device} | epochs={args.epochs}", flush=True)
    print(f"Training class counts [HC, PD, DD]={args.training_class_counts} | "
          f"CE class weights={args.class_weights}", flush=True)
    history = []
    best_macro_f1, best_epoch, best_state, best_validation = float("-inf"), None, None, None
    log_handle = (run_dir / "epochs.jsonl").open("w", encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(3, dtype=torch.float64, device=device)
        predicted_counts = torch.zeros(3, dtype=torch.int64, device=device)
        polarity_correct = torch.zeros((), dtype=torch.int64, device=device)
        question_correct = torch.zeros(30, device=device)
        question_counts = torch.zeros(30, device=device)
        count = 0
        correct = torch.zeros((), dtype=torch.int64, device=device)
        with tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}",
                  unit="batch", dynamic_ncols=True, file=sys.stdout) as progress:
            for step, (x, y, targets) in enumerate(progress):
                x = x.to(device, non_blocking=True).float()
                y = y.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with amp_context(device, dtype):
                    logits, features = model(x, answers[targets])
                    cls_loss = weighted_classification_loss(logits, y, class_weights, args.label_smoothing)
                # Backpropagate through the decoder when either auxiliary objective is enabled.
                with torch.set_grad_enabled(auxiliary_enabled):
                    mask = questionnaire_mask(len(y), args.questionnaire_mask_ratio, mask_generator, device)
                    reconstruction = model.reconstruct(features, conditions, answers[targets], mask)
                    reconstruction_target = targets_bank[targets]
                    auxiliary_loss = masked_reconstruction_loss(reconstruction, reconstruction_target, mask)
                loss = cls_loss
                if args.reconstruction_weight != 0:
                    loss = loss + args.reconstruction_weight * auxiliary_loss
                with torch.no_grad():
                    distances = (reconstruction[:, :, None] - prototypes[None]).square().sum(-1)
                    matches = (distances.argmin(-1) == answers[targets]) & mask
                    polarity_correct += matches.sum()
                    question_correct += matches.sum(0)
                    question_counts += mask.sum(0)
                    predicted_counts += torch.bincount(logits.argmax(-1), minlength=3)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}")
                # Weight by subjects, including the last short microbatch/window.
                window_start = (step // accumulation) * accumulation
                subjects_in_window = min(args.effective_batch_size,
                                         len(datasets["train"]) - window_start * args.micro_batch_size)
                scaler.scale(loss * len(y) / subjects_in_window).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                    scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.gradient_clip)
                    if not torch.isfinite(gradient_norm):
                        raise RuntimeError("Non-finite gradient norm")
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                batch_count = len(y)
                totals += torch.stack((loss.detach(), cls_loss.detach(), auxiliary_loss.detach())).double() * batch_count
                count += batch_count
                correct += (logits.argmax(dim=-1) == y).sum()
                if (step + 1) % args.progress_update_batches == 0 or step + 1 == len(train_loader):
                    display = torch.cat((totals / count, (correct / count).reshape(1))).cpu().tolist()
                    progress.set_postfix(loss=f"{display[0]:.6f}", CE=f"{display[1]:.6f}",
                                         MSE=f"{display[2]:.6f}", accuracy=f"{display[3]:.2%}", refresh=False)
                del logits, features, reconstruction, reconstruction_target, distances, loss, cls_loss, auxiliary_loss, x, y, targets
        # One compact transfer of detached epoch statistics, with no retained graph.
        summary = torch.cat((totals, correct.reshape(1), polarity_correct.reshape(1),
                             predicted_counts, question_correct)).cpu().tolist()
        totals = summary[:3]
        correct, polarity_correct = int(summary[3]), int(summary[4])
        predicted_counts = [int(value) for value in summary[5:8]]
        # Per-question denominators count hidden entries for masked reconstruction.
        counts = question_counts.cpu().tolist()
        per_question_accuracy = [value / n if n else None for value, n in zip(summary[8:], counts)]
        val_metrics, _ = evaluate(model, eval_loaders["val"], device, dtype, args.label_smoothing, class_weights, phase_answers.get("val"), conditions, masked_evaluations.get("val"))
        record = {"epoch": epoch, "train_loss": float(totals[0] / count),
                  "train_classification_loss": float(totals[1] / count),
                  "train_reconstruction_loss": float(totals[2] / count),
                  "train_accuracy": correct / count, "validation": val_metrics,
                  "train_predicted_counts": predicted_counts,
                  "train_polarity_accuracy": polarity_correct / sum(counts),
                  "train_per_question_counts": counts,
                  "train_per_question_accuracy": per_question_accuracy,
                  "learning_rates": {g["name"]: g["lr"] for g in optimizer.param_groups},
                  "seconds": time.time() - epoch_started}
        history.append(record)
        log_handle.write(json.dumps(record, allow_nan=False) + "\n")
        log_handle.flush()
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1, best_epoch, best_validation = float(val_metrics["macro_f1"]), epoch, val_metrics
            best_state = trainable_snapshot(model)
            if args.save_model:
                write_checkpoint(run_dir / "model_trainable.pt", args, best_state, epoch, val_metrics)
            print(f"New best validation macro-F1 {best_macro_f1:.6f} at epoch {epoch}"
                  + (" (checkpoint saved)" if args.save_model else ""), flush=True)
        write_json(run_dir / "run_state.json", {"status": "training", "completed_epochs": epoch,
                   "total_epochs": args.epochs, "elapsed_seconds": time.time() - started,
                   "best_epoch": best_epoch, "best_macro_f1": best_macro_f1,
                   "best_checkpoint_saved": bool(args.save_model),
                   "latest_epoch": record, "model_saved": False})
        emit("epoch_complete", **record)
    log_handle.close()
    # Release the auxiliary target bank; test questionnaires remain in phase_answers.
    del answers, prototypes, targets_bank, optimizer, scheduler, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # The test set is scored by the best validation macro-F1 epoch, not the last one.
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        del best_state
    test_metrics, prediction_data = evaluate(model, eval_loaders["test"], device, dtype, args.label_smoothing, class_weights, phase_answers.get("test"), conditions, masked_evaluations.get("test"))
    write_json(run_dir / "test_predictions.json", {"subject_ids": splits["test"], **prediction_data})
    import csv
    with (run_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "true_label", "predicted_class", "probability_0", "probability_1", "probability_2"])
        for row in zip(splits["test"], prediction_data["true_labels"], prediction_data["predictions"],
                       prediction_data["probabilities"]):
            writer.writerow([row[0], row[1], row[2], *row[3]])
    # model_trainable.pt already holds the best validation macro-F1 epoch.
    result = {"status": "completed", "completed_epochs": args.epochs,
              "model_saved": args.save_model, "auxiliary_enabled": auxiliary_enabled,
              "best_epoch": best_epoch, "best_macro_f1": best_macro_f1,
              "best_validation": best_validation,
              "selection": "best validation macro-F1 epoch; test uses that epoch's weights",
              "test_uses_text": False,
              "test_uses_questionnaire": True,
              "reconstruction_weight": args.reconstruction_weight,
              "test": test_metrics, "final_epoch": history[-1],
              "elapsed_seconds": time.time() - started,
              "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None}
    write_json(run_dir / "results.json", result)
    write_json(run_dir / "run_state.json", result)
    emit("training_complete", **result)


def prepare_text_cache(args):
    cache = Path(args.text_cache)
    if cache.is_file() and not args.reencode_text:
        load_cache(cache, PADS_dataset(args, "train").subject_ids)
        return
    if cache.name != "reconstruction_features.npz":
        raise ValueError("Expected reconstruction_features.npz cache filename")
    subprocess.run([sys.executable, "-m", "reconstruction.prepare_text",
                    "--input", args.statement_json, "--output", str(cache.parent),
                    "--model", args.text_model, "--device", args.device,
                    "--batch-size", str(args.text_batch_size), "--max-length", str(args.text_max_length)], cwd=str(ROOT), check=True)


def main():
    args = SimpleNamespace(**CONFIG)
    if not math.isfinite(args.questionnaire_mask_ratio) or not 0 < args.questionnaire_mask_ratio < 1:
        raise ValueError("questionnaire_mask_ratio must be between zero and one")
    if args.progress_update_batches <= 0:
        raise ValueError("progress_update_batches must be positive")
    if args.epochs <= 0 or args.num_workers < 0 or args.text_batch_size <= 0 or args.text_max_length <= 0:
        raise ValueError("epochs and text encoding sizes must be positive; num_workers must be nonnegative")
    if not (0 < args.val_ratio < 1 and 0 < args.test_ratio < 1 and args.val_ratio + args.test_ratio < 1):
        raise ValueError("validation/test fractions must be positive and sum to less than one")
    if any(value <= 0 for value in (args.lora_lr, args.head_lr, args.decoder_lr, args.adam_eps,
                                    args.gradient_clip, args.decoder_hidden_dim, args.decoder_norm_eps,
                                    args.lora_r, args.lora_alpha, args.torch_num_threads, args.head_channel_dim,
                                    args.fusion_time_dim, args.fusion_question_dim, args.fusion_hidden_dim)):
        raise ValueError("learning rates, dimensions, epsilons and gradient clip must be positive")
    if len(args.adam_betas) != 2 or any(not 0 <= value < 1 for value in args.adam_betas):
        raise ValueError("Adam betas must contain two values in [0, 1)")
    if any(not 0 <= value < 1 for value in (args.lora_dropout, args.head_dropout,
                                           args.patch_dropout, args.backbone_dropout)):
        raise ValueError("dropout probabilities must be in [0, 1)")
    if not 0 <= args.label_smoothing <= 1 or args.reconstruction_weight < 0 or args.weight_decay < 0 or args.decoder_init_std < 0:
        raise ValueError("invalid loss weight, label smoothing, weight decay or initialization scale")
    if not 0 <= args.min_lr <= min(args.lora_lr, args.head_lr, args.decoder_lr):
        raise ValueError("min_lr must be nonnegative and no greater than any initial learning rate")
    if args.precision not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError("precision must be auto, float32, float16 or bfloat16")
    if args.micro_batch_size <= 0 or args.effective_batch_size < args.micro_batch_size or args.effective_batch_size % args.micro_batch_size:
        raise ValueError("effective_batch_size must be a positive multiple of micro_batch_size")
    run_dir = Path(args.run_dir)
    if any((run_dir / name).exists() for name in ("epochs.jsonl", "results.json", "model_trainable.pt")):
        raise FileExistsError(f"Existing run output: {run_dir}; choose a new run_dir")
    args.device = str(resolve_device(args.device))
    resolve_dtype(torch.device(args.device), args.precision)
    prepare_text_cache(args)
    while True:
        try:
            run(args)
            return
        except torch.cuda.OutOfMemoryError:
            if args.micro_batch_size == 1:
                raise
            emit("oom_retry", previous_micro_batch_size=args.micro_batch_size,
                 next_micro_batch_size=1, restart_from_pretrained=True)
            args.micro_batch_size = 1
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
