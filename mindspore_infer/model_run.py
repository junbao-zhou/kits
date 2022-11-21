import os
from dataset import DataLoader
from typing import Any, Callable, Dict, List, Tuple
from utils import AverageMeter
import mindspore
from mindspore import nn
import mindspore.ops
from glob import glob
import numpy as np


def model_run(
    data_loader: DataLoader,
    model: nn.Cell,
    criterions: Dict[str, Tuple[Callable[[mindspore.Tensor, mindspore.Tensor], mindspore.Tensor], float]],
    metrics: Dict[str, Callable[[mindspore.Tensor, mindspore.Tensor, mindspore.Tensor], Any]],
    print_every: int = 0
):

    total_batch_num = len(data_loader)
    epoch_total_loss = AverageMeter()
    epoch_losses = {c: AverageMeter() for c in criterions}
    epoch_metric_s = {m: AverageMeter() for m in metrics}

    for batch, (X, y_true) in enumerate(data_loader):
        batch_size = y_true.shape[0]

        # Forward pass
        predict = model(X)
        probabilities = mindspore.ops.Argmax(axis=1)(predict)
        batch_losses = {
            loss_name: loss_func(predict, y_true)
            for loss_name, (loss_func, _) in criterions.items()
        }
        batch_total_loss = mindspore.Tensor([0.0])
        for loss_name, (_, loss_factor) in criterions.items():
            batch_total_loss += loss_factor * batch_losses[loss_name]
        for loss_name, batch_loss in batch_losses.items():
            epoch_losses[loss_name].update(batch_loss.item(), batch_size)
        epoch_total_loss.update(batch_total_loss.item(), batch_size)

        batch_metric_s = {
            m: metrics[m](predict, probabilities, y_true)
            for m in metrics
        }
        for m, batch_metric in batch_metric_s.items():
            epoch_metric_s[m].update(batch_metric, batch_size)

        if (print_every > 0) and ((batch+1) % print_every == 0):
            losses_str = [f"{name} : {loss:.4f}" for name,
                          loss in batch_losses.items()]
            metrics_str = [f"{name} : {metric}" for name,
                           metric in batch_metric_s.items()]
            print(
                f"""[{batch} / {total_batch_num}] | \
loss : {batch_total_loss} | \
{" | ".join(losses_str)} | \
{" | ".join(metrics_str)} | \
""")

    return epoch_total_loss.get(), {name: meter.get() for name, meter in epoch_metric_s.items()}


def validate(
    valid_loader: DataLoader,
    model: nn.Cell,
    criterions: Dict[str, Tuple[Callable, float]],
    metrics: Dict[str, Callable],
    print_every: int = 0,
):
    print("=" * 40)
    print(f"validate begin")
    valid_loss, valid_metrics = model_run(
        data_loader=valid_loader,
        model=model,
        criterions=criterions,
        metrics=metrics,
        print_every=print_every,
    )
    print("=" * 40)
    print(f"validate end")
    print(
        f'Valid loss: {valid_loss}\t'
        f'Valid metrics: {valid_metrics}')

    return valid_loss, valid_metrics


def test(data_loaders: List[DataLoader], model: nn.Cell, output_path: str):
    for d, data_loader in enumerate(data_loaders):
        print(f"data_loader: {d}")
        np_output_list = []
        for batch, X in enumerate(data_loader):
            print(f"batch {batch}  X.shape {X.shape}")
            predict = model(X)
            probabilities = mindspore.ops.Argmax(axis=1)(predict)
            probabilities_np = probabilities.asnumpy()
            np_output_list.append(probabilities_np)
        np_output = np.concatenate(np_output_list, axis=0)
        print(f"np_output.shape {np_output.shape}")
        np.save(
            os.path.join(output_path, data_loader.dataset.np_file_name),
            np_output,
            allow_pickle=True,
        )

