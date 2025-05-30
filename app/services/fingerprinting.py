import os
import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader


from PIL import Image
from io import BytesIO
from app.utils.dataset import InMemoryDataset
from app.models.stegastamp import StegaStampEncoder, StegaStampDecoder

class FingerprintService:
    def __init__(self, encoder_path, decoder_path, device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        state_dict = torch.load(encoder_path, map_location=self.device)
    
        self.fingerprint_size = state_dict["secret_dense.weight"].shape[-1]

        self.encoder = StegaStampEncoder(128, 3, fingerprint_size=self.fingerprint_size).to(self.device)
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval()

        self.decoder = StegaStampDecoder(128, 3, fingerprint_size=self.fingerprint_size).to(self.device)
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=self.device))
        self.decoder.eval()

        self.transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor()
        ])

    def embed(
        self,
        image_file: BytesIO,
        seed: int = 0,
        save_path: str = "output.png"
    ):
        try:
            torch.manual_seed(seed)
            image = Image.open(image_file).convert("RGB")
            tensor_img = self.transform(image).unsqueeze(0).to(self.device)

            fingerprint = torch.randint(0, 2, (1, self.fingerprint_size), dtype=torch.float).to(self.device)

            with torch.no_grad():
                fingerprinted_image = self.encoder(fingerprint, tensor_img)
                
                mse_loss = F.mse_loss(fingerprinted_image, tensor_img, reduction='mean')

                detected_fingerprint = self.decoder(fingerprinted_image)
                detected_fingerprint = (detected_fingerprint > 0).long()
                bitwise_accuracy = (detected_fingerprint == fingerprint.long()).float().mean().item()

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(fingerprinted_image.cpu(), save_path)

            list_fingerprint = fingerprint.squeeze().cpu().long().numpy().tolist()
            fingerprint_str = "".join(map(str, list_fingerprint))

            metrics = {
                "mse_loss": mse_loss.item(),
                "bitwise_accuracy": bitwise_accuracy
            }

            return save_path, fingerprint_str, metrics
        
        except Exception as e:
            raise ValueError(f"Terjadi kesalahan saat melakukan embed fingerprint: {str(e)}")
    
    def decode(self, image_file: BytesIO):
        try:
            image = Image.open(image_file).convert("RGB")
            tensor_img = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                fingerprint = self.decoder(tensor_img)
                binary_fp = (fingerprint > 0).long().squeeze().cpu().numpy()

            return "".join(map(str, binary_fp.tolist()))
        
        except Exception as e:
            raise ValueError(f"Terjadi kesalahan saat mendekode fingerprint: {str(e)}")
    
    def embed_multiple(
        self, image_paths: list[BytesIO],
        seed: int = 0
    ):
        try:
            torch.manual_seed(seed)
            BATCH_SIZE = 64
            dataset = InMemoryDataset(image_paths, self.transform)
            
            if len(dataset) == 0:
                raise ValueError("Tidak ada gambar valid dalam daftar input")
            
            data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

            all_outputs = []
            all_fingerprints = []
            mse_losses = []
            bitwise_accuracy = 0
            total_images = 0

            fingerprints = torch.randint(0, 2, (1, self.fingerprint_size), dtype=torch.float).to(self.device)

            for images, indices in data_loader:
                images = images.to(self.device)
                batch_size = images.size(0)
                total_images += batch_size

                fingerprints_batch = fingerprints.expand(batch_size, -1)

                with torch.no_grad():
                    fingerprinted_images = self.encoder(fingerprints_batch, images)

                    mse_loss = F.mse_loss(fingerprinted_images, images, reduction='mean')
                    mse_losses.append(mse_loss.item())

                    detected_fingerprints = self.decoder(fingerprinted_images)
                    detected_fingerprints = (detected_fingerprints > 0).long()
                    bitwise_accuracy += (detected_fingerprints == fingerprints_batch).float().mean(dim=1).sum().item()

                for i in range(fingerprinted_images.size(0)):
                    buffer = BytesIO()
                    try:
                        save_image(fingerprinted_images[i].cpu(), buffer, format="PNG")
                        buffer.seek(0)
                        all_outputs.append(buffer)
                    except Exception as e:
                        raise ValueError(f"Gagal memproses gambar pada indeks {indices[i]}: {str(e)}")
                    
                all_fingerprints.extend([
                    "".join(map(str, f.cpu().long().numpy().tolist()))
                    for f in fingerprints
                ])

            avg_mse_loss = sum(mse_losses) / len(mse_losses) if mse_losses else 0
            avg_bitwise_accuracy = bitwise_accuracy / total_images if total_images > 0 else 0

            metrics = {
                "avg_mse_loss": avg_mse_loss,
                "avg_bitwise_accuracy": avg_bitwise_accuracy
            }

            return all_outputs, all_fingerprints, metrics
        
        except Exception as e:
            raise ValueError(f"Terjadi kesalahan saat memproses gambar: {str(e)}")