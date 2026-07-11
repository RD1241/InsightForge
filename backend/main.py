import os
import sys

# Ensure the backend directory is in the Python search path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import routers
from routers import dataset, forecasting, agent

app = FastAPI(
    title="InsightForge API",
    description="AI-powered Retail Decision Support System Backend",
    version="1.0.0"
)

# Include Routers
app.include_router(dataset.router)
app.include_router(forecasting.router)
app.include_router(agent.router)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local ease of use
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    """
    Simple health check endpoint to verify backend status.
    """
    return {
        "status": "healthy",
        "service": "InsightForge API",
        "version": "1.0.0"
    }

# Mount Frontend static files if the frontend folder exists
# This allows serving the dashboard and API from a single local port
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
else:
    # Create the directory and a dummy index.html to ensure it doesn't fail
    os.makedirs(frontend_dir, exist_ok=True)
    with open(os.path.join(frontend_dir, "index.html"), "w") as f:
        f.write("<h1>InsightForge Frontend Placeholder</h1>")
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
