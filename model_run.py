from torch.utils.data import DataLoader
from torch import nn
from typing import Callable, Dict
from utils import AverageMeter
import torch


def model_run(
    data_loader: DataLoader,
    model: nn.Module,
    criterion,
    metrics: Dict[str, Callable],
    device: torch.device,
    optimizer=None,
    print_every: int = 0
):

    if optimizer:
        model.train()
    else:
        model.eval()

    epoch_loss = AverageMeter('Loss')
    epoch_metric_s = {m: AverageMeter(m) for m in metrics}
    batch_metric_s = {m: AverageMeter(m) for m in metrics}

    for batch, (X, y_true) in enumerate(data_loader):
        print(
            f'batch : {batch} ')
        X = X.to(device)
        y_true = y_true.to(device)
        batch_size = y_true.size(0)

        # Forward pass
        predict = model(X)
        loss = criterion(predict, y_true)
        epoch_loss.update(loss.item(), batch_size)

        for m in metrics:
            batch_metric_s[m] = metrics[m](predict, y_true)
            epoch_metric_s[m].update(batch_metric_s[m], batch_size)

        if (print_every > 0) and ((batch+1) % print_every == 0):
            print(
                f'batch : {batch}, loss = {loss}, ')
            for m in metrics:
                print(
                    f'  {m} = {batch_metric_s[m]} epoch : {epoch_metric_s[m]}')

        if optimizer:
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return epoch_loss, epoch_metric_s


def train_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    criterion,
    metrics: Dict[str, Callable],
    device: torch.device,
    optimizer,
    print_every: int = 0,
):
    print(f"training epoch")
    train_loss, train_metrics = model_run(
        data_loader=train_loader,
        model=model,
        criterion=criterion,
        metrics=metrics,
        device=device,
        optimizer=optimizer,
        print_every=print_every,
    )
    print(
        f'Train loss: {train_loss}\t'
        f'Train metrics: {train_metrics}\t')
    return train_loss, train_metrics


def validate(
    valid_loader: DataLoader,
    model: nn.Module,
    criterion,
    metrics: Dict[str, Callable],
    device: torch.device,
    print_every: int = 0,
):
    with torch.no_grad():
        valid_loss, valid_metrics = model_run(
            data_loader=valid_loader,
            model=model,
            criterion=criterion,
            metrics=metrics,
            device=device,
            print_every=print_every,
        )
    print(
        f'Valid loss: {valid_loss}\t'
        f'Valid metrics: {valid_metrics}')

    return valid_loss, valid_metrics
