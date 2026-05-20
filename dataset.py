import zipfile
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def prepare_local_data(archive_path="data.zip", extract_to="./data"):
    data_dir = Path(extract_to)
    
    if data_dir.exists() and any(data_dir.iterdir()):
        return

    archive_file = Path(archive_path)
    if not archive_file.exists():
        raise FileNotFoundError(f"Файл {archive_path} не найден")

    print(f"Распаковка {archive_path}...")
    with zipfile.ZipFile(archive_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

class MVTecDataset(Dataset):
    def __init__(self, root_dir="./data", categories=("hazelnut",), is_train=True, transform=None):
        self.root_dir = Path(root_dir)
        self.categories = [categories] if isinstance(categories, str) else categories
        self.is_train = is_train
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self._load_paths()

    def _load_paths(self):
        phase = "train" if self.is_train else "test"
        
        for category in self.categories:
            phase_dir = self.root_dir / category / phase
            
            if not phase_dir.exists():
                continue

            for img_path in phase_dir.rglob("*.png"):
                self.image_paths.append(img_path)
                if self.is_train or img_path.parent.name == "good":
                    self.labels.append(0)
                else:
                    self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(root_dir="./data", categories=("hazelnut",), batch_size=32, num_workers=4):
    prepare_local_data()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = MVTecDataset(root_dir, categories, is_train=True, transform=transform)
    test_dataset = MVTecDataset(root_dir, categories, is_train=False, transform=transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, test_loader