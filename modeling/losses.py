import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probability of the true class
        pt = torch.exp(-ce_loss)
        
        # Apply the focusing factor
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            # Ensure alpha is on the same device as the inputs
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
