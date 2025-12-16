import torch.nn as nn

class AssociationPredictor(nn.Module):
    """下游关联预测模型，用于评估特征质量"""
    def __init__(self, feature_dim, n_diseases):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_diseases)
        )
    
    def forward(self, x):
        return self.predictor(x)