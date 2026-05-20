import torch
import json
from pathlib import Path
from dataset import get_dataloaders
from model import ConvAutoencoder, LargeConvAutoencoder
from engine import train_autoencoder, evaluate_model

def export_model(model, save_path, device):
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    torch.onnx.export(
        model, dummy_input, save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

def update_metrics_json(exp_name, score):
    domain_map = {
        "hazelnut": "Орехи (Hazelnut)",
        "leather": "Кожа (Leather)",
        "mixed": "Микс (Смешанный)"
    }
    model_map = {
        "base": "Base (Сверточная сеть)",
        "large": "Large (Увеличенная емкость)"
    }
    
    parts = exp_name.split('_')
    if len(parts) != 2:
        return
    domain_key, model_key = parts[0], parts[1]
    
    domain = domain_map.get(domain_key)
    model_type = model_map.get(model_key)
    
    if not domain or not model_type:
        return
        
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    metrics_file = out_dir / "metrics.json"
    
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
        
    if domain not in data:
        data[domain] = {}
        
    data[domain][model_type] = f"{score:.4f}"
    
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def run_pipeline(name, model_class, categories, epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = get_dataloaders(categories=categories, batch_size=batch_size)
    model = model_class().to(device)
    
    model = train_autoencoder(model, train_loader, num_epochs=epochs, device=device)
    auc = evaluate_model(model, test_loader, device=device)
    
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    onnx_path = out_dir / f"{name}.onnx"
    
    export_model(model, str(onnx_path), device)
    update_metrics_json(name, auc)
    
    return auc

def main():
    experiments = [
        ("hazelnut_base", ConvAutoencoder, ["hazelnut"]),
        ("hazelnut_large", LargeConvAutoencoder, ["hazelnut"]),
        ("leather_base", ConvAutoencoder, ["leather"]),
        ("leather_large", LargeConvAutoencoder, ["leather"]),
        ("mixed_base", ConvAutoencoder, ["hazelnut", "leather"]),
        ("mixed_large", LargeConvAutoencoder, ["hazelnut", "leather"])
    ]
    
    metrics = {}
    for name, arch, cats in experiments:
        print(f"Запуск эксперимента: {name}")
        auc = run_pipeline(name, arch, cats, epochs=50, batch_size=32)
        metrics[name] = auc
        
    print("\nИтоговые результаты:")
    for exp_name, score in metrics.items():
        print(f"{exp_name} | ROC-AUC: {score:.4f}")

if __name__ == "__main__":
    main()