from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import isi, phq9, subtype, psg
from app.routers import correlation

app.include_router(correlation.router, prefix="/correlation", tags=["Correlation"])

app = FastAPI(title="SleepScope Backend API")

# CORS for V0 front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Replace with your frontend URL after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(isi.router, prefix="/isi", tags=["ISI"])
app.include_router(phq9.router, prefix="/phq9", tags=["PHQ-9"])
app.include_router(subtype.router, prefix="/subtype", tags=["Subtype Classification"])
app.include_router(psg.router, prefix="/psg", tags=["PSG Analysis"])

@app.get("/")
def root():
    return {"message": "SleepScope Backend Running Successfully 🚀"}
