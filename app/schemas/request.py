from pydantic import BaseModel, Field
from typing import Optional

class GenerationRequest(BaseModel):
    model_name: str = Field(..., description="Nama model StyleGAN2 yang akan digunakan")
    seed: Optional[int] = Field(None, description="Seed untuk random generator")
    truncation_psi: float = Field(0.7, description="Nilai truncation psi (0-1)", ge=0, le=1)