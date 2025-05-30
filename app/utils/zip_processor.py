import os
import zipfile
import tempfile
import json
from io import BytesIO

class ZipImageProcessor:
    @staticmethod
    def extract_images(
        zip_file: BytesIO,
        tmp_dir: str
    ):
        try:
            zip_path = os.path.join(tmp_dir, "input.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file.read())

            extracted_path = os.path.join(tmp_dir, "extracted")
            os.makedirs(extracted_path, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extracted_path)

            image_paths = []
            image_filenames = []
            for root, _, files in os.walk(extracted_path):
                for name in files:
                    if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, name)
                        image_paths.append(file_path)
                        image_filenames.append(name)
                    
            if not image_paths:
                raise ValueError("Tidak ada gambar valid dalam zip")
            
            return image_paths, image_filenames
        
        except zipfile.BadZipFile:
            raise ValueError("File zip tidak valid atau rusak")
        except Exception as e:
            raise ValueError(f"Terjadi kesalahan saat mengekstrak file zip: {str(e)}")

    @staticmethod
    def create_zip(
        images: list[BytesIO],
        filenames: list[str],
        fingerprint_dict: dict,
        output_dir: str,
        request_id: str
    ):
        os.makedirs(output_dir, exist_ok=True)
        zip_filename = os.path.join(output_dir, f"fingerprinted_images_{request_id}.zip")

        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as out_zip:
            for img_buf, name in zip(images, filenames):
                out_zip.writestr(f"images/fingerprinted_{os.path.basename(name)}", img_buf.getvalue())

            out_zip.writestr("fingerprints.json", json.dumps(fingerprint_dict, indent=2))

        return zip_filename
