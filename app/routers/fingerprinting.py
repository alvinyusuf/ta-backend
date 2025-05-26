from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO
from app.services.fingerprinting import FingerprintService

fp_service = FingerprintService(
    encoder_path="pretrained_models/128_encoder.pth",
    decoder_path="pretrained_models/128_decoder.pth"
)

router = APIRouter(
    prefix="/api/fingerprinting",
    tags=["fingerprinting"]
)

@router.post("/embed")
async def embed_fingerprint(image: UploadFile = File(...)):
    image_data = await image.read()
    result = fp_service.embed(BytesIO(image_data), identical_fp=True)

    return StreamingResponse(result, media_type="image/png")

@router.post("/decode")
async def decode_fingerprint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    fingerprint = fp_service.decode(BytesIO(image_bytes))
    return JSONResponse({"fingerprint": fingerprint})