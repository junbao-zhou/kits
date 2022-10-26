import mindspore

class ConfusionMatrix:
    def __init__(self, n_classes, device):
        self.n_classes = n_classes
        self.device = device

    def get_conf_matrix(self, x: mindspore.Tensor, y: mindspore.Tensor):
        x, y = x.detach(), y.detach()
        # sizes should be "batch_size x H x W"
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify
        # idxs are labels and predictions
        idxs = mindspore.ops.Stack([x_row, y_row], dim=0)
        ones = mindspore.ops.Ones((idxs.shape[-1]), device=self.device).long()
        conf = mindspore.ops.Zeros(
            (self.n_classes, self.n_classes), device=self.device).long()
        conf = conf.index_put_(
            tuple(idxs), ones, accumulate=True)
        return conf

    def get_accuracy(self, x, y, eps=1e-6):
        conf = self.get_conf_matrix(x, y).double()
        tp = conf.diag()
        total = conf.sum(dim=1) + eps
        return tp / total  # returns "acc mean"


def dice(predicts: mindspore.Tensor, labels: mindspore.Tensor, eps=1e-6):
    
    intersect = (mindspore.LogicalAnd(predicts>0, labels>0)).sum()
    union = predicts.sum() + labels.sum()
    dice = (2.0 * intersect.double()) / (union.double() + eps)
    return dice


def dices(probabilities: mindspore.Tensor, labels: mindspore.Tensor):
    probabilities, labels = probabilities.detach(), labels.detach()
    tk_dice = dice(probabilities > 0, labels > 0)
    tu_dice = dice(probabilities > 1, labels > 1)
    return mindspore.Tensor([tk_dice, tu_dice])
