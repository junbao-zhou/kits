import os
from torch.utils.data import DataLoader
from torch import nn
from typing import Any, Callable, Dict, List, Tuple
from dataset import SegmentKits19
from utils import AverageMeter
import torch
import numpy as np


def model_run(
    data_loader: DataLoader,
    model: nn.Module,
    criterions: Dict[str, Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], float]],
    metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any]],
    device: torch.device,
    optimizer=None,
    print_every: int = 0
):

    if optimizer:
        model.train()
    else:
        model.eval()

    total_batch_num = len(data_loader)
    epoch_total_loss = AverageMeter()
    epoch_losses = {c: AverageMeter() for c in criterions}
    epoch_metric_s = {m: AverageMeter() for m in metrics}

    for batch, (X, y_true) in enumerate(data_loader):
        X = X.to(device)
        y_true = y_true.to(device)
        batch_size = y_true.size(0)

        # Forward pass
        predict = model(X)
        probabilities = predict.argmax(dim=1)
        batch_losses = {
            loss_name: loss_func(predict, y_true)
            for loss_name, (loss_func, _) in criterions.items()
        }
        batch_total_loss = torch.tensor(0.0, device=device)
        for loss_name, (_, loss_factor) in criterions.items():
            batch_total_loss += loss_factor * batch_losses[loss_name]
        with torch.no_grad():
            for loss_name, batch_loss in batch_losses.items():
                epoch_losses[loss_name].update(
                    batch_loss.detach().item(), batch_size)
            epoch_total_loss.update(
                batch_total_loss.detach().item(), batch_size)

        with torch.no_grad():
            batch_metric_s = {
                m: metrics[m](predict, probabilities, y_true)
                for m in metrics
            }
            for m, batch_metric in batch_metric_s.items():
                epoch_metric_s[m].update(batch_metric, batch_size)

        if (print_every > 0) and ((batch+1) % print_every == 0):
            losses_str = [
                f"{name} : {loss:.4f}" for name, loss in batch_losses.items()]
            metrics_str = [
                f"{name} : {metric}" for name, metric in batch_metric_s.items()]
            print(
                f"""[{batch} / {total_batch_num}] | \
loss : {batch_total_loss:.4f} | \
{" | ".join(losses_str)} | \
{" | ".join(metrics_str)} | \
""")

        if optimizer:
            # Backward pass
            optimizer.zero_grad()
            batch_total_loss.backward()
            optimizer.step()

    return epoch_total_loss.get(), {name: meter.get() for name, meter in epoch_metric_s.items()}


def train_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    criterions: Dict[str, Tuple[Callable, float]],
    metrics: Dict[str, Callable],
    device: torch.device,
    optimizer,
    print_every: int = 0,
):
    print("=" * 40)
    print(f"training epoch begin")
    train_loss, train_metrics = model_run(
        data_loader=train_loader,
        model=model,
        criterions=criterions,
        metrics=metrics,
        device=device,
        optimizer=optimizer,
        print_every=print_every,
    )
    print("=" * 40)
    print(f"training epoch end")
    print(
        f'Train loss: {train_loss}\t'
        f'Train metrics: {train_metrics}\t')
    return train_loss, train_metrics


def validate(
    valid_loader: DataLoader,
    model: nn.Module,
    criterions: Dict[str, Tuple[Callable, float]],
    metrics: Dict[str, Callable],
    device: torch.device,
    print_every: int = 0,
):
    print("=" * 40)
    print(f"validate begin")
    with torch.no_grad():
        valid_loss, valid_metrics = model_run(
            data_loader=valid_loader,
            model=model,
            criterions=criterions,
            metrics=metrics,
            device=device,
            print_every=print_every,
        )
    print("=" * 40)
    print(f"validate end")
    print(
        f'Valid loss: {valid_loss}\t'
        f'Valid metrics: {valid_metrics}')

    return valid_loss, valid_metrics


def test_epoch(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
):
    output_list = []
    true_list = []
    for batch, (X, y_true) in enumerate(data_loader):
        print(f"{batch = }")
        X = X.to(device)
        y_true = y_true.to(device)
        batch_size = y_true.size(0)

        predict = model(X)
        probabilities = predict.argmax(dim=1)

        output_list.append(probabilities)
        true_list.append(y_true)
    output = torch.concat(output_list, dim=0)
    trues = torch.concat(true_list, dim=0)
    return output, trues


def test(
    cases: List[int],
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any]],
    output_dir: str,
):
    with torch.no_grad():
        for case in cases:
            print(f"{case = }")
            data_loader = DataLoader(
                dataset=SegmentKits19([case]),
                batch_size=batch_size,
                num_workers=6,
            )
            output, trues = test_epoch(
                data_loader,
                model,
                device,
            )
            metric_s = {
                m: metrics[m](None, output, trues)
                for m in metrics
            }
            metrics_str = [
                f"{name} : {metric}" for name, metric in metric_s.items()]
            print(" | ".join(metrics_str))

            output_np = output.detach().cpu().numpy()
            np.save(
                os.path.join(output_dir, f"test_{case}.nii.gz.npy"),
                output_np,
                allow_pickle=True,
            )


def test_from_file(
    cases: List[int],
    batch_size: int,
    metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any]],
    output_dir: str,
):
    with torch.no_grad():
        for case in cases:
            print(f"{case = }")
            data_loader = DataLoader(
                dataset=SegmentKits19([case]),
                batch_size=batch_size,
                num_workers=6,
            )
            true_list = []
            for batch, (X, y_true) in enumerate(data_loader):
                print(f"{batch = }")
                true_list.append(y_true)
            trues = torch.concat(true_list, dim=0)
            print(f"{trues.shape = }")
            output = np.load(
                os.path.join(output_dir, f"test_{case}.nii.gz.npy"),
                allow_pickle=True,
            )
            output = torch.from_numpy(output)
            metric_s = {
                m: metrics[m](None, output, trues)
                for m in metrics
            }
            metrics_str = [
                f"{name} : {metric}" for name, metric in metric_s.items()]
            print(" | ".join(metrics_str))