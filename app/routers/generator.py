from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import uuid
import os

from app.models.stylegan import generate_image
from app.schemas.request import GenerationRequest

router = APIRouter(
    prefix="/api/generator",
    tags=["generator"]
)

@router.post("/generate")
async def generate(request: GenerationRequest):
    try:
        # Buat nama file unik
        filename = f"{uuid.uuid4()}.png"
        save_path = os.path.join("static", "images", filename)
        
        # Generate gambar di background task
        await generate_image(
            model_name=request.model_name,
            seed=request.seed,
            truncation_psi=request.truncation_psi,
            class_idx=request.class_idx,
            noise_mode=request.noise_mode,
            save_path=save_path
        )
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "Gambar sedang digenerate",
                "image_url": f"/static/images/{filename}",
                "request_id": str(uuid.uuid4()),
                "seed": request.seed
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models():
    # Dapatkan daftar model StyleGAN2 yang tersedia
    models_dir = "pretrained_models"
    models = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    
    return {"models": models}