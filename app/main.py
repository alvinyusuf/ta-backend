from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.routers import generator

app = FastAPI(
    title="StyleGAN2 Generator API",
    description="API untuk generate gambar menggunakan model StyleGAN2",
    version="1.0.0"
)

# Middleware untuk CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain frontend Anda untuk production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder untuk menyimpan dan menyajikan gambar hasil
app.mount("/static", StaticFiles(directory="static"), name="static")

# Router
app.include_router(generator.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)