import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss,_Loss
class WeightedMSELoss(_WeightedLoss):
    __constants__ = ["reduction"]

    def forward(self, input: torch.tensor, target: torch.tensor) -> torch.Tensor:

        sample_weights = self.weight[input.to(torch.int32)]
        return nn.functional.mse_loss(input, target, reduction=self.reduction,weight = sample_weights)


#https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289
class FocalLossCELoss(nn.CrossEntropyLoss):
    __constants__ = ["ignore_index", "reduction", "label_smoothing","gamma"]
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        gamma: int = 2
    ) -> None:
        super().__init__(weight, size_average,ignore_index, reduce, reduction,label_smoothing)
        self.gamma = gamma
    

    def forward(self, input: torch.tensor, target: torch.tensor) -> torch.Tensor:
        ce_loss = torch.nn.functional.cross_entropy(input, target,weight = None,ignore_index = self.ignore_index,label_smoothing = self.label_smoothing, reduction='none') 
        pt = torch.exp(-ce_loss)
        #instead of defining another function just doing it here
        focal_loss = ( (1-pt)**self.gamma * ce_loss)
        if self.reduction == "none":
            return focal_loss
        elif self.reduction == "sum":
            return torch.sum(focal_loss)
        elif self.reduction == "mean":
            return focal_loss.mean()
            #not sure if i should do by weights again if already doing in cross_entropy, not even sure if they should be used in first place? 
            #also this needs to be per sample weight sum, not this 
            #return torch.sum(focal_loss) / torch.sum(self.weight)
        else:
            raise ValueError(
                f"Invalid reduction mode: {self.reduction}. Expected one of 'none', 'mean', 'sum'."
            )