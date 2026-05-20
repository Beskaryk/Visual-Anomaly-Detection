import streamlit as st
import onnxruntime as ort
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.ndimage import gaussian_filter
import json

st.set_page_config(page_title="Industrial Anomaly Detection", layout="wide")

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

@st.cache_resource
def get_device_and_extractor():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = PaDiMExtractor(device)
    return device, extractor

@st.cache_resource
def load_all_models():
    models_dir = Path("models")
    loaded_models = {}
    if not models_dir.exists():
        return loaded_models
        
    for file_path in models_dir.glob("*"):
        name = file_path.stem
        if file_path.suffix == ".onnx":
            loaded_models[name] = {
                "type": "autoencoder",
                "session": ort.InferenceSession(str(file_path), providers=['CPUExecutionProvider'])
            }
        elif file_path.suffix == ".pt" and "onnx" not in file_path.suffixes:
            data = torch.load(file_path, map_location="cpu", weights_only=True)
            loaded_models[name] = {
                "type": "padim",
                "mean": data['mean'],
                "inv_cov": data['inv_cov']
            }
    return loaded_models

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((256, 256))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    return np.expand_dims(img_array, axis=0)

def generate_heatmap(anomaly_map: np.ndarray) -> np.ndarray:
    res_min, res_max = anomaly_map.min(), anomaly_map.max()
    normalized_map = (anomaly_map - res_min) / (res_max - res_min + 1e-8)
    heatmap = plt.cm.jet(normalized_map)[:, :, :3]
    return (heatmap * 255).astype(np.uint8)

def analyze_defect(original_image: Image.Image, anomaly_map: np.ndarray, model_type: str):
    annotated_image = original_image.resize((256, 256)).copy()
    
    if model_type == "autoencoder":
        threshold = 0.12 
        mask = anomaly_map > threshold
        is_defective = np.sum(mask) > 30
        
    elif model_type == "padim":
        mean_val = np.mean(anomaly_map)
        std_val = np.std(anomaly_map)
        
        threshold = mean_val + 3.0 * std_val
        mask = anomaly_map > threshold
        
        is_defective = np.sum(mask) > 50
    
    if is_defective and np.any(mask):
        y_indices, x_indices = np.where(mask)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        draw = ImageDraw.Draw(annotated_image)
        padding = 5
        draw.rectangle(
            [max(0, x_min-padding), max(0, y_min-padding), 
             min(256, x_max+padding), min(256, y_max+padding)], 
            outline="red", width=3
        )
            
    return annotated_image, is_defective

def compute_autoencoder(session, input_tensor: np.ndarray):
    input_name = session.get_inputs()[0].name
    reconstructed = session.run(None, {input_name: input_tensor})[0]
    
    residual = np.abs(input_tensor - reconstructed)
    residual_map = np.mean(residual, axis=1).squeeze()
    
    residual_map = gaussian_filter(residual_map, sigma=1)
    
    heatmap = generate_heatmap(residual_map)
    reconstructed_img = np.transpose(reconstructed.squeeze(), (1, 2, 0))
    reconstructed_img = (reconstructed_img * 255).astype(np.uint8)
    
    return heatmap, reconstructed_img, residual_map

def compute_padim(extractor, mean, inv_cov, input_tensor: np.ndarray, device):
    tensor = torch.from_numpy(input_tensor).to(device)
    mean = mean.to(device)
    inv_cov = inv_cov.to(device)
    
    with torch.no_grad():
        features = extractor(tensor)
        B, C, H, W = features.shape
        features = features.view(B, C, H * W).permute(0, 2, 1)
        
        diff = features - mean.unsqueeze(0)
        diff_ic = torch.einsum('bic,icc->bic', diff, inv_cov)
        dist = torch.sum(diff * diff_ic, dim=2)
        dist = torch.sqrt(dist).view(B, 1, H, W)
        
        dist = F.interpolate(dist, size=(256, 256), mode='bilinear', align_corners=False)
        anomaly_map = dist.squeeze().cpu().numpy()
        
    anomaly_map = gaussian_filter(anomaly_map, sigma=3)
    heatmap = generate_heatmap(anomaly_map)
    
    return heatmap, None, anomaly_map

def main():
    st.title("Система промышленного контроля качества")
    
    device, padim_extractor = get_device_and_extractor()
    models = load_all_models()
    
    if not models:
        st.error("В директории models/ не найдены рабочие модели.")
        return
        
    st.sidebar.header("Параметры инференса")
    selected_model_name = st.sidebar.selectbox("Архитектура анализа", sorted(list(models.keys())))
    uploaded_file = st.sidebar.file_uploader("Загрузка тестового образца", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
        input_tensor = preprocess(original_image)
        config = models[selected_model_name]
        
        st.markdown(f"### Анализ: `{selected_model_name}`")
        
        with st.spinner("Выполнение вычислений..."):
            if config["type"] == "autoencoder":
                heatmap, reconstructed, raw_map = compute_autoencoder(config["session"], input_tensor)
            elif config["type"] == "padim":
                heatmap, reconstructed, raw_map = compute_padim(padim_extractor, config["mean"], config["inv_cov"], input_tensor, device)
            
            annotated_img, is_defective = analyze_defect(original_image, raw_map, config["type"])
        
        if is_defective:
            st.error("Результат: Обнаружен дефект (превышение порога допустимой ошибки).")
        else:
            st.success("Результат: Норма (отклонения в пределах допустимой погрешности).")
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(annotated_img, caption="Оригинал (Разметка зоны брака)", use_container_width=True)
        with col2:
            st.image(heatmap, caption="Пространственная тепловая карта", use_container_width=True)
        with col3:
            if reconstructed is not None:
                st.image(reconstructed, caption="Топология реконструкции (Автоэнкодер)", use_container_width=True)
            else:
                st.info("Архитектура PaDiM не предполагает пиксельной реконструкции.")
                
    st.divider()
    st.subheader("Сводная матрица метрик (ROC-AUC)")
    
    metrics_file = Path("models/metrics.json")
    
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
            
        df = pd.DataFrame(metrics_data).T
        df.index.name = "Домен данных"
        df.reset_index(inplace=True)
        
        cols = ["Домен данных"] + sorted([c for c in df.columns if c != "Домен данных"])
        df = df[cols]
        
        st.table(df)
    else:
        st.warning("Файл с результатами тестирования (models/metrics.json) не найден. Запустите пайплайн оценки для генерации артефакта.")

if __name__ == "__main__":
    main()