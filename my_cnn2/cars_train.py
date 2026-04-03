import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cars_dataset import get_cars_datasets
from cars_models import (
    get_model,
    set_last_layer_only,
    set_full_finetune,
    set_gradual_unfreeze,
    set_sliding_window_finetune,
)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy_topk(logits, targets, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results.append((correct_k / targets.size(0)).item())
        return results


def evaluate(model, dataloader, device, criterion):
    model.eval()

    running_loss = 0.0
    n_samples = 0
    all_top1 = 0.0
    all_top5 = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            top1, top5 = accuracy_topk(logits, labels, topk=(1, 5))

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            all_top1 += top1 * batch_size
            all_top5 += top5 * batch_size
            n_samples += batch_size

    return {
        "loss": running_loss / max(1, n_samples),
        "accuracy_top1": all_top1 / max(1, n_samples),
        "accuracy_top5": all_top5 / max(1, n_samples),
        "n_samples": n_samples,
    }


def get_optimizer(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)


def main():
    parser = argparse.ArgumentParser(description="Train Stanford Cars transfer-learning models")
    parser.add_argument("--cars-root", type=Path, default=Path("stanford_cars"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-name", type=str, choices=["densenet121", "resnet152"], required=True)
    parser.add_argument("--strategy", type=str, choices=["scratch", "last_layer", "full", "gradual", "sliding_window"], required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--unfreeze-every", type=int, default=10)
    parser.add_argument("--slide-every", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Load full train/test datasets
    train_dataset_full, test_dataset = get_cars_datasets(args.cars_root, image_size=224)

    indices = np.arange(len(train_dataset_full))
    labels = np.array([train_dataset_full.samples[i]["label"] for i in indices])

    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_fraction,
        random_state=args.seed,
        stratify=labels,
    )

    train_dataset = Subset(train_dataset_full, train_idx)
    val_dataset = Subset(train_dataset_full, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # pretrained = args.strategy != "scratch"
    # model = get_model(
    #     model_name=args.model_name,
    #     num_classes=196,
    #     pretrained=pretrained,
    # ).to(device)

    if args.strategy == "scratch":
        set_full_finetune(model)
        strategy_note = "all parameters trainable from scratch"

    elif args.strategy == "last_layer":
        set_last_layer_only(model)
        strategy_note = "only final classification layer trainable"

    elif args.strategy == "full":
        set_full_finetune(model)
        strategy_note = "all pretrained parameters trainable"

    elif args.strategy == "gradual":
        info = set_gradual_unfreeze(
            model,
            epoch=epoch,
            unfreeze_every=args.unfreeze_every,
        )
        strategy_note = (
            f"head + {info['n_backbone_blocks_trainable']} output-side backbone block(s) trainable"
        )

    elif args.strategy == "sliding_window":
        info = set_sliding_window_finetune(
            model,
            epoch=epoch,
            slide_every=args.slide_every,
            window_size=args.window_size,
        )
        strategy_note = (
            f"head always trainable; sliding window over backbone blocks "
            f"{info['window_start']} to {info['window_end']}"
        )

    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    criterion = nn.CrossEntropyLoss()

    run_name = f"{args.model_name}_cars_{args.strategy}"
    best_model_path = args.output_dir / f"{run_name}_best.pth"
    final_model_path = args.output_dir / f"{run_name}_final.pth"
    metrics_path = args.output_dir / f"{run_name}_metrics.json"

    history = []
    best_val_top1 = -1.0

    for epoch in range(1, args.epochs + 1):
        # Strategy setup each epoch
        if args.strategy == "scratch":
            set_full_finetune(model)
            strategy_note = "all parameters trainable from scratch"
        elif args.strategy == "last_layer":
            set_last_layer_only(model)
            strategy_note = "only final classification layer trainable"
        elif args.strategy == "full":
            set_full_finetune(model)
            strategy_note = "all pretrained parameters trainable"
        elif args.strategy == "gradual":
            n_unfrozen = set_gradual_unfreeze(model, epoch, unfreeze_every=args.unfreeze_every)
            strategy_note = f"{n_unfrozen} output-side blocks trainable"
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")

        optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

        model.train()
        running_loss = 0.0
        n_train = 0
        train_top1_total = 0.0
        train_top5_total = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            top1, top5 = accuracy_topk(logits, labels, topk=(1, 5))

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            train_top1_total += top1 * batch_size
            train_top5_total += top5 * batch_size
            n_train += batch_size

        train_loss = running_loss / max(1, n_train)
        train_top1 = train_top1_total / max(1, n_train)
        train_top5 = train_top5_total / max(1, n_train)

        val_metrics = evaluate(model, val_loader, device, criterion)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "train_accuracy_top1": float(train_top1),
            "train_accuracy_top5": float(train_top5),
            "val_accuracy_top1": float(val_metrics["accuracy_top1"]),
            "val_accuracy_top5": float(val_metrics["accuracy_top5"]),
            "strategy": args.strategy,
            "strategy_note": strategy_note,
            "n_train_samples": int(n_train),
            "n_val_samples": int(val_metrics["n_samples"]),
        }
        history.append(row)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  train_loss:      {train_loss:.6f}")
        print(f"  val_loss:        {val_metrics['loss']:.6f}")
        print(f"  train_top1:      {train_top1:.4f}")
        print(f"  train_top5:      {train_top5:.4f}")
        print(f"  val_top1:        {val_metrics['accuracy_top1']:.4f}")
        print(f"  val_top5:        {val_metrics['accuracy_top5']:.4f}")
        print(f"  strategy_note:   {strategy_note}")

        if val_metrics["accuracy_top1"] > best_val_top1:
            best_val_top1 = val_metrics["accuracy_top1"]
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model -> {best_model_path}")

        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)

    torch.save(model.state_dict(), final_model_path)
    print(f"\nSaved final model -> {final_model_path}")

    # Test evaluation using best checkpoint
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device, criterion)

    history.append({
        "epoch": "test",
        "loss": float(test_metrics["loss"]),
        "accuracy_top1": float(test_metrics["accuracy_top1"]),
        "accuracy_top5": float(test_metrics["accuracy_top5"]),
        "n_samples": int(test_metrics["n_samples"]),
    })

    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n=== TEST RESULTS ===")
    print(f"  test_loss:  {test_metrics['loss']:.6f}")
    print(f"  test_top1:  {test_metrics['accuracy_top1']:.4f}")
    print(f"  test_top5:  {test_metrics['accuracy_top5']:.4f}")
    print(f"Metrics saved -> {metrics_path}")


if __name__ == "__main__":
    main()