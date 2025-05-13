import os
import base64
from PIL import Image
from io import BytesIO

def image_to_base64(image_path):
    """Mengubah gambar menjadi string base64"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def save_base64_image(base64_string, save_path):
    """Menyimpan string base64 sebagai gambar"""
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    
    # Buat direktori jika belum ada
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    img.save(save_path)
    return save_path