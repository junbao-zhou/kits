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

    def get_accuracy(self, x, y):
        conf = self.get_conf_matrix(x, y).double()
        tp = conf.diag()
        total = conf.sum(dim=1)
        return tp /  total # returns "acc mean"


def dices(probabilities: torch.Tensor, labels: torch.Tensor):
    try:
        # Compute tumor+kidney Dice
        tk_pd = probabilities > 0
        tk_gt = labels > 0
        tk_dice = 2*torch.logical_and(tk_pd, tk_gt).sum()/(
            tk_pd.sum() + tk_gt.sum()
        )
    except ZeroDivisionError:
        return torch.Tensor([0.0, 0.0])

    try:
        # Compute tumor Dice
        tu_pd = probabilities > 1
        tu_gt = labels > 1
        tu_dice = 2*torch.logical_and(tu_pd, tu_gt).sum()/(
            tu_pd.sum() + tu_gt.sum()
        )
    except ZeroDivisionError:
        return torch.Tensor([tk_dice, 0.0])

    return torch.Tensor([tk_dice, tu_dice])