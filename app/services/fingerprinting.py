import os
import torch
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from io import BytesIO
from app.utils.dataset import InMemoryDataset
from app.models.stegastamp import StegaStampEncoder, StegaStampDecoder

class FingerprintService:
    def __init__(self, encoder_path, decoder_path=None, device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load encoder model
        state_dict = torch.load(encoder_path, map_location=self.device)

        self.fingerprint_size = state_dict["secret_dense.weight"].shape[-1]

        self.encoder = StegaStampEncoder(128, 3, fingerprint_size=self.fingerprint_size).to(self.device)
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval()

        # Load decoder model (optional)
        self.decoder = None
        if decoder_path:
            self.decoder = StegaStampDecoder(128, 3, fingerprint_size=self.fingerprint_size).to(self.device)
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
            self.decoder.eval()

        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor()
        ])

    def embed(self, image_file: BytesIO):
        image = Image.open(image_file).convert("RGB")
        tensor_img = self.transform(image).unsqueeze(0).to(self.device)

        fingerprint = torch.randint(0, 2, (1, self.fingerprint_size), dtype=torch.float).to(self.device)

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
    
    def embed_multiple(self, image_files: list[BytesIO], seed: int = 0):
        torch.manual_seed(seed)
        BATCH_SIZE = 64
        dataset = InMemoryDataset(image_files, self.transform)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        all_outputs = []
        all_fingerprints = []

        fingerprints = torch.randint(0, 2, (1, self.fingerprint_size), dtype=torch.float).to(self.device)
        
        for images, indices in data_loader:
            images = images.to(self.device)

            fingerprints = fingerprints.expand(images.size(0), -1)

            with torch.no_grad():
                fingerprinted_images = self.encoder(fingerprints, images)

            for i in range(fingerprinted_images.size(0)):
                buffer = BytesIO()
                save_image(fingerprinted_images[i].cpu(), buffer, format="PNG")
                buffer.seek(0)
                all_outputs.append(buffer)
                
            all_fingerprints.extend([
                 "".join(map(str, f.cpu().long().numpy().tolist()))
                for f in fingerprints
            ])

        return all_outputs, all_fingerprints