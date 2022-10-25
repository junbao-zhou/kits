import torch


class ConfusionMatrix:
    def __init__(self, n_classes, device):
        self.n_classes = n_classes
        self.device = device
    #     self.reset()

    # def reset(self):
    #     self.conf_matrix = torch.zeros(
    #         (self.n_classes, self.n_classes), device=self.device).long()
    #     self.ones = None
    #     self.last_scan_size = None  # for when variable scan size is used

    # def addBatch(self, x, y):  # x=preds, y=targets

    #     # sizes should be "batch_size x H x W"
    #     x_row = x.reshape(-1)  # de-batchify
    #     y_row = y.reshape(-1)  # de-batchify

    #     # idxs are labels and predictions
    #     idxs = torch.stack([x_row, y_row], dim=0)

    #     # ones is what I want to add to conf when I
    #     if self.ones is None or self.last_scan_size != idxs.shape[-1]:
    #         self.ones = torch.ones((idxs.shape[-1]), device=self.device).long()
    #         self.last_scan_size = idxs.shape[-1]

    #     # make confusion matrix (cols = gt, rows = pred)
    #     self.conf_matrix = self.conf_matrix.index_put_(
    #         tuple(idxs), self.ones, accumulate=True)

    def get_conf_matrix(self, x: torch.Tensor, y: torch.Tensor):
        x, y = x.detach(), y.detach()
        # sizes should be "batch_size x H x W"
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify
        # idxs are labels and predictions
        idxs = torch.stack([x_row, y_row], dim=0)
        ones = torch.ones((idxs.shape[-1]), device=self.device).long()
        conf = torch.zeros(
            (self.n_classes, self.n_classes), device=self.device).long()
        conf = conf.index_put_(
            tuple(idxs), ones, accumulate=True)
        return conf

    def get_accuracy(self, x, y, eps=1e-6):
        conf = self.get_conf_matrix(x, y).double()
        tp = conf.diag()
        total = conf.sum(dim=1) + eps
        return tp / total  # returns "acc mean"


def dice(predicts: torch.Tensor, labels: torch.Tensor, eps=1e-6):
    intersect = (torch.logical_and(predicts, labels)).sum()
    union = predicts.sum() + labels.sum()
    dice = (2.0 * intersect.double()) / (union.double() + eps)
    return dice


def dices(probabilities: torch.Tensor, labels: torch.Tensor):
    probabilities, labels = probabilities.detach(), labels.detach()
    tk_dice = dice(probabilities > 0, labels > 0)
    tu_dice = dice(probabilities > 1, labels > 1)
    return torch.Tensor([tk_dice, tu_dice])
