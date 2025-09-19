# app/main.py
from fastapi import FastAPI
from app.api.routes import router as api_router

app = FastAPI(title="Ayla Hair AI")
app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API Ayla Hair AI"}
