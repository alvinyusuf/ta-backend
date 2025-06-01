from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO
import uuid
import os
import tempfile

from app.services.fingerprinting import FingerprintService
from app.utils.zip_processor import ZipImageProcessor
from app.utils.response import success_response, error_response

fp_service = FingerprintService(
    encoder_path="pretrained_models/128_encoder.pth",
    decoder_path="pretrained_models/128_decoder.pth"
)

router = APIRouter(
    prefix="/api/fingerprinting",
    tags=["fingerprinting"]
)

@router.post("/embed")
async def embed_fingerprint(image: UploadFile = File(...), seed: int = Form(...)):
    try:
        image_data = await image.read()
        filename = f"{uuid.uuid4()}.png"
        save_path = os.path.join("static", "images", "embed", filename)

        _, fingerprint_str, metrics = fp_service.embed(BytesIO(image_data), seed=seed, save_path=save_path)

        return success_response(
            message="Fingerprint embedded successfully",
            data={
                "image_url": f"/static/images/embed/{filename}",
                "filename": filename,
                "fingerprint": fingerprint_str,
                "metrics": metrics,
                "request_id": str(uuid.uuid4()),
            }
        )

    except Exception as e:
        return error_response(
            message="Error embedding fingerprint",
            error=str(e),
            status_code=500
        )

@router.post("/decode")
async def decode_fingerprint(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        fingerprint = fp_service.decode(BytesIO(image_bytes))

        return success_response(
            message="Fingerprint decoded successfully",
            data={
                "fingerprint": fingerprint
            }
        )

    except Exception as e:
        return error_response(
            message="Error decoding fingerprint",
            error=str(e),
            status_code=500
        )

@router.post("/embed-batch")
async def embed_fingerprint_batch(
    file: UploadFile = File(...),
    seed: int = Form(...)
):
    try:
        if not file.filename.lower().endswith('.zip'):
            raise ValueError("File must be a zip archive containing images")
        
        request_id = str(uuid.uuid4())
        save_path = os.path.join("static", "zip", "batch_fingerprints")

        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_file = BytesIO(await file.read())
            image_paths, filenames = ZipImageProcessor.extract_images(zip_file, tmp_dir)

            outputs, fingerprints, metrics = fp_service.embed_batch(image_paths, seed)

            fingerprint_dict = {
                "fingerprint": fingerprints[0],
            }

            zip_filename = ZipImageProcessor.create_zip(
                outputs,
                filenames,
                fingerprint_dict,
                output_dir=save_path,
                request_id=request_id
            )

        return success_response(
            message="Batch fingerprint embedding completed successfully",
            data={
                "zip_url": f"/{zip_filename}",
                "request_id": request_id,
                "metrics": metrics,
                "fingerprints": fingerprint_dict
            }
        )
    
    except Exception as e:
        return error_response(
            message="Error embedding fingerprints in batch",
            error=str(e),
            status_code=500
        )
