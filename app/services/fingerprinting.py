import os
import torch
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from io import BytesIO
from app.models.stegastamp import StegaStampEncoder, StegaStampDecoder

FINGERPRINT_SIZE = 100  # default, will be overwritten

class FingerprintService:
    def __init__(self, encoder_path, decoder_path=None, device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load encoder model
        state_dict = torch.load(encoder_path, map_location=self.device)
        global FINGERPRINT_SIZE
        FINGERPRINT_SIZE = state_dict["secret_dense.weight"].shape[-1]

        self.encoder = StegaStampEncoder(128, 3, fingerprint_size=FINGERPRINT_SIZE).to(self.device)
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval()

        # Load decoder model (optional)
        self.decoder = None
        if decoder_path:
            self.decoder = StegaStampDecoder(128, 3, fingerprint_size=FINGERPRINT_SIZE).to(self.device)
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
            self.decoder.eval()

        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor()
        ])

    def embed(self, image_file: BytesIO, identical_fp=True):
        image = Image.open(image_file).convert("RGB")
        tensor_img = self.transform(image).unsqueeze(0).to(self.device)

        if identical_fp:
            fingerprint = torch.randint(0, 2, (1, FINGERPRINT_SIZE), dtype=torch.float).to(self.device)
        else:
            fingerprint = torch.randint(0, 2, (1, FINGERPRINT_SIZE), dtype=torch.float).to(self.device)

        with torch.no_grad():
            fingerprinted_image = self.encoder(fingerprint, tensor_img)

        # Convert output tensor to image bytes
        output_buffer = BytesIO()
        save_image(fingerprinted_image.cpu(), output_buffer, format="PNG")
        output_buffer.seek(0)

        print(f"Fingerprint size: {str(fingerprint)}")

        return output_buffer
    
    def decode(self, image_file: BytesIO):
        if self.decoder is None:
            raise ValueError("Decoder model not loaded.")

        image = Image.open(image_file).convert("RGB")
        tensor_img = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            fingerprint = self.decoder(tensor_img)
            binary_fp = (fingerprint > 0).long().squeeze().cpu().numpy()

        return "".join(map(str, binary_fp.tolist()))
