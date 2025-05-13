import os
import numpy as np
import pickle
from PIL import Image
import torch

# Path ke model StyleGAN2 yang sudah dilatih
MODELS_DIR = "pretrained_models"

# Fungsi untuk load model
def load_model(model_name):
    model_path = os.path.join(MODELS_DIR, model_name)
    
    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    
    return G

# Fungsi untuk generate gambar
async def generate_image(model_name, seed=None, truncation_psi=0.7, save_path="output.png"):
    try:
        # Load model
        G = load_model(model_name)
        
        # Set seed jika diberikan
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        
        # Set random state
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate latent vector
        z = torch.randn(1, G.z_dim).cuda()
        
        # Generate image
        with torch.no_grad():
            img = G(z, truncation_psi=truncation_psi)
        
        # Konversi ke format yang dapat disimpan
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].cpu().numpy()
        
        # Simpan gambar
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(img).save(save_path)
        
        return save_path
    
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise e