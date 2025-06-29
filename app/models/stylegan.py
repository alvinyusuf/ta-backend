import os
import numpy as np
import PIL.Image
import torch
import dnnlib
import legacy
from typing import Optional

MODELS_DIR = "pretrained_models"

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with dnnlib.util.open_url(model_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    return G

async def generate_image(
    model_name: str, 
    seed: Optional[int] = None, 
    # truncation_psi: float = 0.7, 
    save_path: str = "output.png"
):
    try:
        truncation_psi = 0.7  # Default value, can be adjusted as needed
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_path = os.path.join(MODELS_DIR, model_name)
        G = load_model(model_path)
        
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        
        label = None
        if G.c_dim != 0:
            label = torch.zeros([1, G.c_dim], device=device)
            label[:, 0] = 1

        noise_mode = 'const'
        
        print(f'Generating image for seed {seed} with truncation_psi={truncation_psi}...')
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img[0].cpu().numpy()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        PIL.Image.fromarray(img, 'RGB').save(save_path)
        
        return save_path
    
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise e