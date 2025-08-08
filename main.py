from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import predict

app = FastAPI(title="Digital Health Intelligence API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)

@app.get("/")
def root():
    return {"message": "✅ Backend is running successfully!"}
