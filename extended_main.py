import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from dataset import get_dataloaders
from sklearn.metrics import roc_auc_score
from pathlib import Path
import json

class PaDiMExtractor:
    def __init__(self, device):
        self.device = device
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        self.model.eval()
        self.features = []
        
        def hook(module, input, output):
            self.features.append(output)
            
        self.model.layer1.register_forward_hook(hook)
        self.model.layer2.register_forward_hook(hook)
        self.model.layer3.register_forward_hook(hook)

    def __call__(self, x):
        self.features = []
        with torch.no_grad():
            self.model(x)
            
        feat1, feat2, feat3 = self.features
        feat2 = F.interpolate(feat2, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        feat3 = F.interpolate(feat3, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        
        return torch.cat([feat1, feat2, feat3], dim=1)

def train_padim(train_loader, device):
    extractor = PaDiMExtractor(device)
    feature_list = []
    
    for images, _ in train_loader:
        images = images.to(device)
        features = extractor(images)
        feature_list.append(features.cpu())
        
    all_features = torch.cat(feature_list, dim=0)
    N, C, H, W = all_features.shape
    
    all_features = all_features.view(N, C, H * W).permute(2, 1, 0)
    mean = torch.mean(all_features, dim=2)
    
    cov = torch.zeros(H * W, C, C)
    I = torch.eye(C)
    
    for i in range(H * W):
        diff = all_features[i] - mean[i].unsqueeze(1)
        cov[i] = (diff @ diff.T) / (N - 1) + 0.01 * I
        
    inv_cov = torch.linalg.inv(cov)
    return mean, inv_cov, extractor

def evaluate_padim(mean, inv_cov, extractor, test_loader, device):
    all_scores = []
    all_labels = []
    
    mean = mean.to(device)
    inv_cov = inv_cov.to(device)
    
    for images, labels in test_loader:
        images = images.to(device)
        features = extractor(images)
        B, C, H, W = features.shape
        
        features = features.view(B, C, H * W).permute(0, 2, 1)
        diff = features - mean.unsqueeze(0)
        
        diff_ic = torch.einsum('bic,icc->bic', diff, inv_cov)
        dist = torch.sum(diff * diff_ic, dim=2)
        dist = torch.sqrt(dist)
        
        image_scores = torch.max(dist, dim=1)[0].cpu().numpy()
        
        all_scores.extend(image_scores)
        all_labels.extend(labels.numpy())
        
    auc = roc_auc_score(all_labels, all_scores)
    return auc

def run_extended_pipeline(name, categories, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(categories=categories, batch_size=batch_size)
    
    mean, inv_cov, extractor = train_padim(train_loader, device)
    auc = evaluate_padim(mean, inv_cov, extractor, test_loader, device)
    
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    torch.save({'mean': mean.cpu(), 'inv_cov': inv_cov.cpu()}, out_dir / f"{name}.pt")
    
    domain_map = {
        "hazelnut": "Орехи (Hazelnut)",
        "leather": "Кожа (Leather)",
        "mixed": "Микс (Смешанный)"
    }
    
    metrics_file = out_dir / "metrics.json"
    data = {}
    
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pass
        
    domain = domain_map[name.split('_')[0]]
    if domain not in data:
        data[domain] = {}
    data[domain]["Expanded (Словарь PaDiM)"] = f"{auc:.4f}"
    
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    return auc

def main():
    experiments = [
        ("hazelnut_extended", ["hazelnut"]),
        ("leather_extended", ["leather"]),
        ("mixed_extended", ["hazelnut", "leather"])
    ]
    
    metrics = {}
    for name, cats in experiments:
        print(f"Запуск эксперимента: {name}")
        auc = run_extended_pipeline(name, cats, batch_size=32)
        metrics[name] = auc
        
    print("\nИтоговые результаты:")
    for exp_name, score in metrics.items():
        print(f"{exp_name} | ROC-AUC: {score:.4f}")

if __name__ == "__main__":
    main()