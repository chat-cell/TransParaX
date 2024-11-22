import torch
import torch.nn as nn
import torch.nn.functional as F

class ParameterExtractionLoss(nn.Module):
    def __init__(self, param_weights=None, uncertainty_weight=0.1):
        super().__init__()
        self.param_weights = param_weights
        self.uncertainty_weight = uncertainty_weight
        
    def forward(self, pred_dict, target):
        mean = pred_dict['mean']
        variance = pred_dict['variance']
        
        if self.param_weights is not None:
            mse_loss = torch.mean(self.param_weights * (mean - target)**2)
        else:
            mse_loss = F.mse_loss(mean, target)
        
        uncertainty_loss = 0.5 * (torch.log(variance) + (target - mean)**2 / variance).mean()
        
        total_loss = mse_loss + self.uncertainty_weight * uncertainty_loss
        
        return total_loss, {
            'mse_loss': mse_loss.item(),
            'uncertainty_loss': uncertainty_loss.item()
        }

class Semi_ParameterExtractionLoss(nn.Module):
    def __init__(self, 
                 param_weights=None,
                 uncertainty_weight=0.1,
                 pseudo_label_weight=0.1,
                 tau_base=0.8,
                 gamma=0.5,
                 alpha=0.01,
                 eta=0.1,
                 beta=1.0):
        """
        Args:
            param_weights: Optional weights for different parameters
            uncertainty_weight: Weight for uncertainty loss component (λ2)
            pseudo_label_weight: Weight for pseudo-label loss component (λ1)
            tau_base: Base threshold for pseudo-labeling
            gamma: Control parameter for threshold decay
            alpha: Rate of threshold decay
            eta: Validation accuracy impact factor
            beta: Uncertainty penalty factor
        """
        super().__init__()
        self.param_weights = param_weights
        self.uncertainty_weight = uncertainty_weight
        self.pseudo_label_weight = pseudo_label_weight
        self.tau_base = tau_base
        self.gamma = gamma
        self.alpha = alpha
        self.eta = eta
        self.beta = beta
        
    def compute_supervised_loss(self, pred_mean, pred_var, target):
        """Compute supervised loss with parameter prediction and uncertainty estimation"""
        if self.param_weights is not None:
            param_loss = torch.mean(self.param_weights * (pred_mean - target)**2)
        else:
            param_loss = F.mse_loss(pred_mean, target)
            
        # Uncertainty loss as per equation (20)
        uncertainty_loss = torch.mean(torch.log(pred_var) + (target - pred_mean)**2 / pred_var)
        
        return param_loss, 0.5 * uncertainty_loss
    
    def compute_adaptive_threshold(self, current_iter, val_accuracy):
        """Compute adaptive threshold τt as per equation (15)"""
        decay_factor = 1 + self.gamma * math.exp(-self.alpha * current_iter)
        accuracy_factor = 1 - self.eta * val_accuracy
        return self.tau_base * decay_factor * accuracy_factor
    
    def compute_pseudo_label_loss(self, pred_mean, pred_var, pseudo_labels, threshold):
        """Compute pseudo-label loss for unlabeled data"""
        # Compute confidence scores based on prediction uncertainty
        confidence_scores = torch.exp(-self.beta * torch.sum(pred_var, dim=1))
        
        # Select high-confidence samples using threshold
        mask = confidence_scores > threshold
        
        if not mask.any():
            return torch.tensor(0.0, device=pred_mean.device)
            
        # Compute weighted MSE loss for selected samples as per equation (18)
        selected_loss = torch.mean(
            confidence_scores[mask].unsqueeze(1) * 
            (pred_mean[mask] - pseudo_labels[mask])**2
        )
        
        return selected_loss
        
    def forward(self, pred_dict, target_dict):
        """
        Args:
            pred_dict: Dictionary containing:
                - 'labeled': {'mean': tensor, 'variance': tensor}
                - 'unlabeled': {'mean': tensor, 'variance': tensor}
                - 'pseudo_labels': tensor of pseudo-labels for unlabeled data
            target_dict: Dictionary containing:
                - 'labeled': tensor of true labels
                - 'current_iter': current training iteration
                - 'val_accuracy': current validation accuracy
        """
        # Compute supervised loss components
        param_loss, uncertainty_loss = self.compute_supervised_loss(
            pred_dict['labeled']['mean'],
            pred_dict['labeled']['variance'],
            target_dict['labeled']
        )
        
        supervised_loss = param_loss + self.uncertainty_weight * uncertainty_loss
        
        # Compute adaptive threshold
        threshold = self.compute_adaptive_threshold(
            target_dict['current_iter'],
            target_dict['val_accuracy']
        )
        
        # Compute pseudo-label loss if unlabeled data exists
        pseudo_loss = 0.0
        if 'unlabeled' in pred_dict:
            pseudo_loss = self.compute_pseudo_label_loss(
                pred_dict['unlabeled']['mean'],
                pred_dict['unlabeled']['variance'],
                pred_dict['pseudo_labels'],
                threshold
            )
        
        # Total loss as per equation (17)
        total_loss = supervised_loss + self.pseudo_label_weight * pseudo_loss
        
        return total_loss, {
            'param_loss': param_loss.item(),
            'uncertainty_loss': uncertainty_loss.item(),
            'pseudo_loss': pseudo_loss.item() if isinstance(pseudo_loss, torch.Tensor) else pseudo_loss,
            'threshold': threshold
        }