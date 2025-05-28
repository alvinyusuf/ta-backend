import os
import zipfile
import tempfile
import json
from io import BytesIO

class ZipImageProcessor:
    @staticmethod
    def extract_images(zip_file: BytesIO):
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "input.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file.read())

            extracted_path = os.path.join(temp_dir, "extracted")
            os.makedirs(extracted_path, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extracted_path)

            image_buffers = []
            image_filenames = []

            for root, _, files in os.walk(extracted_path):
                for name in files:
                    file_path = os.path.join(root, name)
                    with open(file_path, "rb") as img_file:
                        image_buffers.append(BytesIO(img_file.read()))
                        image_filenames.append(name)

            return image_buffers, image_filenames

    @staticmethod
    def create_zip(images: list[BytesIO], filenames: list[str], fingerprint_dict: dict):
        """Buat zip output yang berisi gambar dan file JSON fingerprint"""
        output_zip_buffer = BytesIO()
        with zipfile.ZipFile(output_zip_buffer, "w", zipfile.ZIP_DEFLATED) as out_zip:
            for img_buf, name in zip(images, filenames):
                out_zip.writestr(f"images/{name}", img_buf.getvalue())

            out_zip.writestr("fingerprints.json", json.dumps(fingerprint_dict, indent=2))

        output_zip_buffer.seek(0)
        return output_zip_buffer
