from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import uuid
import os

from app.models.stylegan import generate_image
from app.schemas.request import GenerationRequest
from app.utils.response import success_response, error_response

router = APIRouter(
    prefix="/api/generator",
    tags=["generator"]
)

@router.post("/generate")
async def generate(request: GenerationRequest):
    try:
        filename = f"{uuid.uuid4()}.png"
        save_path = os.path.join("static", "images", "GAN", filename)
        
        await generate_image(
            model_name=request.model_name,
            seed=request.seed,
            truncation_psi=request.truncation_psi,
            class_idx=request.class_idx,
            noise_mode=request.noise_mode,
            save_path=save_path
        )

        return success_response(
            message="Gambar berhasil digenerate",
            data={
                "image_url": f"/static/images/GAN/{filename}",
                "filename": filename,
                "request_id": str(uuid.uuid4()),
            }
        )
    
    except Exception as e:
        return error_response(
            message="Terjadi kesalahan saat menggenerate gambar",
            detail=str(e),
            status_code=500
        )

@router.get("/models")
async def list_models():
    try:
        models_dir = "pretrained_models"
        
        if not os.path.exists(models_dir):
            raise FileNotFoundError(
                "Direktori model tidak ditemukan"
            )
        
        models = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]

        return success_response(
            message="Daftar model berhasil diambil",
            data={"models": models}
        )
        
    except Exception as e:
        return error_response(
            message="Terjadi kesalahan saat mengambil daftar model",
            detail=str(e),
            status_code=500
        )
