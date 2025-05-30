from torch.utils.data import Dataset
from PIL import Image
from typing import List

class InMemoryDataset(Dataset):
    def __init__(self, image_paths: List[str], transform):
        self.image_paths = []
        self.transform = transform
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    img.convert("RGB").verify()
                self.image_paths.append(path)
            except Exception as e:
                print(f"Peringatan: Melewati gambar tidak valid: {str(e)}")
        if not self.image_paths:
            raise ValueError("Tidak ada gambar valid dalam daftar input")

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            return self.transform(img), idx
        except Exception as e:
            raise ValueError(f"Gagal memproses gambar pada indeks {idx}: {str(e)}")

    def __len__(self):
        return len(self.image_paths)