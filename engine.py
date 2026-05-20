import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device)
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:9]).eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        features_x = self.feature_extractor(x)
        features_y = self.feature_extractor(y)
        return self.criterion(features_x, features_y)

def train_autoencoder(model, train_loader, num_epochs=50, lr=1e-3, device=None, use_expanded_loss=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = model.to(device)
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    
    if use_expanded_loss:
        perceptual_criterion = PerceptualLoss(device)
        
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            
            if use_expanded_loss:
                loss = 0.2 * l1_criterion(outputs, images) + 0.8 * perceptual_criterion(outputs, images)
            else:
                loss = mse_criterion(outputs, images)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch [{epoch+1:03d}/{num_epochs:03d}] | Loss: {epoch_loss:.6f}")
        
    return model

def evaluate_model(model, test_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, images)
            scores = loss.mean(dim=[1, 2, 3]).cpu().numpy()
            
            all_scores.extend(scores)
            all_labels.extend(labels.numpy())
            
    auc = roc_auc_score(all_labels, all_scores)
    return auc