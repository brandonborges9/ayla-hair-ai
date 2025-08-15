from fastapi import FastAPI, File, UploadFile
from app.api.routes import router as api_router

app = FastAPI(title="Ayla Hair AI")

app.include_router(api_router, prefix="/api")