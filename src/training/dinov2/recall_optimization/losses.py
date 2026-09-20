# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class FocalLoss(nn.Module):
#     """
#     Multi-class Focal Loss for PyTorch classification.

#     Focal Loss:
#         FL(pt) = - alpha_t * (1 - pt)^gamma * log(pt)

#     In this implementation:
#         - inputs: logits with shape [batch_size, num_classes]
#         - targets: class indices with shape [batch_size]
#         - alpha: None or list/tensor of class weights
#           Example for binary classification:
#               alpha=[0.25, 0.75]
#               class 0 = benign, class 1 = malignant
#     """

#     def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
#         super().__init__()
#         self.gamma = gamma
#         self.reduction = reduction

#         if alpha is None:
#             self.alpha = None
#         else:
#             self.alpha = torch.tensor(alpha, dtype=torch.float32)

#     def forward(self, inputs, targets):
#         ce_loss = F.cross_entropy(inputs, targets, reduction="none")
#         pt = torch.exp(-ce_loss)

#         focal_loss = (1.0 - pt) ** self.gamma * ce_loss

#         if self.alpha is not None:
#             alpha = self.alpha.to(inputs.device)
#             alpha_t = alpha.gather(0, targets)
#             focal_loss = alpha_t * focal_loss

#         if self.reduction == "mean":
#             return focal_loss.mean()
#         elif self.reduction == "sum":
#             return focal_loss.sum()
#         elif self.reduction == "none":
#             return focal_loss
#         else:
#             raise ValueError(f"Unsupported reduction: {self.reduction}")
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    inputs: logits [B, C]
    targets: labels [B]
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss

        raise ValueError(f"Unsupported reduction: {self.reduction}")


def compute_class_balanced_weights(samples_per_class, beta=0.999):
    """
    Class-Balanced weights based on effective number of samples.

    effective_num = 1 - beta^n
    weight = (1 - beta) / effective_num

    Then normalize so that sum(weights) = num_classes.
    """

    samples = torch.tensor(samples_per_class, dtype=torch.float32)

    if torch.any(samples <= 0):
        raise ValueError(f"All class sample counts must be positive. Got: {samples_per_class}")

    beta = float(beta)

    effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), samples)
    weights = (1.0 - beta) / effective_num

    weights = weights / weights.sum() * len(samples_per_class)

    return weights


class ClassBalancedCrossEntropyLoss(nn.Module):
    """
    Class-Balanced Softmax Cross Entropy.

    This is CrossEntropyLoss with effective-number-based class weights.
    """

    def __init__(self, samples_per_class, beta=0.999, reduction="mean"):
        super().__init__()
        self.samples_per_class = samples_per_class
        self.beta = float(beta)
        self.reduction = reduction

        weights = compute_class_balanced_weights(samples_per_class, beta)
        self.register_buffer("weights", weights)

    def forward(self, inputs, targets):
        weights = self.weights.to(inputs.device)
        return F.cross_entropy(inputs, targets, weight=weights, reduction=self.reduction)


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss for 2-class / multi-class softmax classification.

    For each sample:
        focal_loss = (1 - pt)^gamma * CE
        cb_weight = weight[target]
        final_loss = cb_weight * focal_loss
    """

    def __init__(self, samples_per_class, beta=0.999, gamma=0.5, reduction="mean"):
        super().__init__()
        self.samples_per_class = samples_per_class
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.reduction = reduction

        weights = compute_class_balanced_weights(samples_per_class, beta)
        self.register_buffer("weights", weights)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        weights = self.weights.to(inputs.device)
        cb_weight = weights.gather(0, targets)

        loss = cb_weight * focal_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss

        raise ValueError(f"Unsupported reduction: {self.reduction}")