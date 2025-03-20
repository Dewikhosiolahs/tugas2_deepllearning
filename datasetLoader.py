import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

class ImageTensorDataset(Dataset):
    def __init__(self, dataset_path, size=(224, 224)):
        self.image_tensors = []
        self.labels = []
        self.class_to_idx = {}  # Mapping class ke indeks
        current_label = 0
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),  # Konversi ke tensor [0, 1]
        ])
        for person in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person)
            if not os.path.isdir(person_path):
                continue
            if person not in self.class_to_idx:
                self.class_to_idx[person] = current_label
                current_label += 1
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(person_path, img_file)
                    img = Image.open(img_path).convert("RGB")  # Baca gambar
                    img_tensor = transform(img)  # Resize & convert ke tensor
                    self.image_tensors.append(img_tensor)
                    self.labels.append(self.class_to_idx[person])
        self.image_tensors = torch.stack(self.image_tensors)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        return self.image_tensors[idx], self.labels[idx]