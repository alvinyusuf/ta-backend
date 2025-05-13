import os
import numpy as np
import PIL.Image
import torch
import dnnlib
import legacy
from typing import Optional

# Path ke model StyleGAN2 yang sudah dilatih
MODELS_DIR = "pretrained_models"

# Fungsi untuk load model
def load_model(model_path):
    """Load model StyleGAN2 dari path file pkl"""
    print(f'Loading network from "{model_path}"...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with dnnlib.util.open_url(model_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    return G

# Fungsi untuk generate gambar
async def generate_image(
    model_name: str, 
    seed: Optional[int] = None, 
    truncation_psi: float = 0.7, 
    class_idx: Optional[int] = None,
    noise_mode: str = 'const',
    save_path: str = "output.png"
):
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        model_path = os.path.join(MODELS_DIR, model_name)
        G = load_model(model_path)
        
        # Set seed jika diberikan
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        
        # Generate latent vector z
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        
        # Prepare label (class conditioning)
        label = torch.zeros([1, G.c_dim], device=device)
        if G.c_dim != 0:
            if class_idx is not None:
                label[:, class_idx] = 1
            else:
                # Default ke kelas pertama jika tidak dispesifikkan
                label[:, 0] = 1
        
        # Generate image
        print(f'Generating image for seed {seed} with truncation_psi={truncation_psi}...')
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        
        # Konversi ke format yang dapat disimpan
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].cpu().numpy()
        
        # Simpan gambar
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        PIL.Image.fromarray(img, 'RGB').save(save_path)
        
        return save_path
    
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise e