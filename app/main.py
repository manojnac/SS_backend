from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Routers
from app.routers import isi, phq9, subtype, psg, correlation

app = FastAPI(title="SleepScope API", version="1.0.0")

# CORS — allow Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can later restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(isi.router, prefix="/isi", tags=["ISI"])
app.include_router(phq9.router, prefix="/phq9", tags=["PHQ-9"])
app.include_router(subtype.router, prefix="/subtype", tags=["Subtype Classification"])
app.include_router(psg.router, prefix="/psg", tags=["PSG Analysis"])
app.include_router(correlation.router, prefix="/correlation", tags=["Correlation"])

@app.get("/")
def root():
    return {"message": "SleepScope backend running successfully!"}
