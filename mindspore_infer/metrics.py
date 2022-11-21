import mindspore
import mindspore.ops
import mindspore.numpy

class ConfusionMatrix:
    def __init__(self, n_classes):
        self.n_classes = n_classes
    #     self.reset()


    def get_conf_matrix(self, x: mindspore.Tensor, y: mindspore.Tensor):
        # x, y = x.detach(), y.detach()
        # sizes should be "batch_size x H x W"
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify
        # idxs are labels and predictions
        idxs = mindspore.ops.Stack(axis=0)([x_row, y_row])
        ones = mindspore.ops.Ones()((idxs.shape[-1]), mindspore.int32)
        conf = mindspore.ops.Zeros()(
            (self.n_classes, self.n_classes), mindspore.int32)
        conf[tuple(idxs)] = ones
        return conf

    def get_accuracy(self, x: mindspore.Tensor, y: mindspore.Tensor, eps=1e-6):
        acc_list = []
        cast = mindspore.ops.Cast()
        for cls in range(self.n_classes):
            tp = mindspore.ops.LogicalAnd()((x == cls), (y == cls)).sum()
            total = (x == cls).sum()
            acc_list.append(float(tp) / (float(total) + eps))
        return mindspore.Tensor(acc_list)


def dice(predicts: mindspore.Tensor, labels: mindspore.Tensor, eps=1e-6):
    intersect = (mindspore.ops.LogicalAnd()(predicts, labels)).sum()
    union = predicts.sum() + labels.sum()
    cast =  mindspore.ops.Cast()
    dice = (2.0 * cast(intersect, mindspore.float32)) / (cast(union, mindspore.float32) + eps)
    return dice


def dices(probabilities: mindspore.Tensor, labels: mindspore.Tensor):
    tk_dice = dice(probabilities > 0, labels > 0)
    tu_dice = dice(probabilities > 1, labels > 1)
    return mindspore.ops.Stack(axis=0)([tk_dice, tu_dice])
