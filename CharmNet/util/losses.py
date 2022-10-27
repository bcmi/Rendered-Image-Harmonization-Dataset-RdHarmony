import torch
import torch.nn as nn

def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims

class Loss(nn.Module):
    def __init__(self, pred_outputs, gt_outputs):
        super().__init__()
        self.pred_outputs = pred_outputs
        self.gt_outputs = gt_outputs


class MSE(Loss):
    def __init__(self, pred_name='images', gt_image_name='target_images'):
        super(MSE, self).__init__(pred_outputs=(pred_name,), gt_outputs=(gt_image_name,))

    def forward(self, pred, label):
        label = label.view(pred.size())
        loss = torch.mean((pred - label) ** 2, dim=get_dims_with_exclusion(label.dim(), 0))
        return loss


class MaskWeightedMSE(Loss):
    def __init__(self, min_area=100.0, pred_name='images',
                 gt_image_name='target_images', gt_mask_name='masks'):
        super(MaskWeightedMSE, self).__init__(pred_outputs=(pred_name, ),
                                              gt_outputs=(gt_image_name, gt_mask_name))
        self.min_area = min_area

    def forward(self, pred, label, mask):
        label = label.view(pred.size())
        reduce_dims = get_dims_with_exclusion(label.dim(), 0)

        loss = (pred - label) ** 2
        delimeter = pred.size(1) * torch.clamp_min(torch.sum(mask, dim=reduce_dims), self.min_area)
        loss = torch.sum(loss, dim=reduce_dims) / delimeter

        return loss

class SoftCrossEntropy(Loss):
    def __init__(self, pred_name='style', gt_style_name='target_style'):
        super(SoftCrossEntropy, self).__init__(pred_outputs=(pred_name,), gt_outputs=(gt_style_name,))

    def forward(self, pred, label):
        pred = -torch.log(pred)
        loss = torch.mean(torch.sum(torch.mul(label, pred), -1))
        return loss
        

class CrossEntropy(Loss):
    def __init__(self, pred_name='style', gt_style_name='target_style'):
        super(CrossEntropy, self).__init__(pred_outputs=(pred_name,), gt_outputs=(gt_style_name,))
        self.NLL = nn.NLLLoss()

    def forward(self, pred, label):
        pred = torch.log(pred)
        loss = self.NLL(pred, label)
        return loss