from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO
from app.services.fingerprinting import FingerprintService
from app.utils.zip_processor import ZipImageProcessor

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
    result = fp_service.embed(BytesIO(image_data))

    return StreamingResponse(result, media_type="image/png")

@router.post("/decode")
async def decode_fingerprint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    fingerprint = fp_service.decode(BytesIO(image_bytes))
    return JSONResponse({"fingerprint": fingerprint})

# @router.post("/embed-batch")
# async def embed_fingerprint_batch(files: list[UploadFile] = File(...), seed: int = Form(...)):
#     image_files = [BytesIO(await file.read()) for file in files]
#     outputs, fingerprints = fp_service.embed_multiple(image_files, seed)
#     return {
#         "count": len(outputs),
#         "fingerprints": fingerprints
#     }

@router.post("/embed-batch")
async def embed_fingerprint_batch(file: UploadFile = File(...), seed: int = Form(...)):
    zip_file = BytesIO(await file.read())

    image_buffers, filenames = ZipImageProcessor.extract_images(zip_file)
    outputs, fingerprints = fp_service.embed_multiple(image_buffers, seed)

    fp_dict = {name: fp for name, fp in zip(filenames, fingerprints)}

    result_zip = ZipImageProcessor.create_zip(outputs, filenames, fp_dict)

    return StreamingResponse(
        result_zip,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=fingerprinted_images.zip"}
)
