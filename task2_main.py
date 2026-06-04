# ============================================================
# TASK 2 - Face Authentication | main.py (FastAPI)
# 
# HOW TO RUN IN GOOGLE COLAB:
# ----------------------------
# 1. Install:
#    !pip install fastapi uvicorn deepface tf-keras python-multipart pyngrok
#
# 2. Start server with ngrok tunnel (for public URL in Colab):
#    from pyngrok import ngrok
#    import subprocess, threading
#
#    def run():
#        subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])
#    threading.Thread(target=run, daemon=True).start()
#
#    public_url = ngrok.connect(8000)
#    print("API URL:", public_url)
#
# 3. Then open: {public_url}/docs  → Swagger UI to test the API
# ============================================================

import os
import uuid
import shutil
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Import our predict function
from task2_predict import load_model_and_predict

# ----------------------------
# Create FastAPI app
# ----------------------------
app = FastAPI(
    title       = "Face Authentication API",
    description = "Verify if two face images belong to the same person using FaceNet.",
    version     = "1.0.0",
)

# Temp folder to store uploaded images
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ----------------------------
# Root endpoint - health check
# ----------------------------
@app.get("/")
def root():
    return {
        "message": "Face Authentication API is running ✅",
        "docs":    "/docs",
        "verify":  "POST /verify"
    }


# ----------------------------
# /verify endpoint
# Accepts two face images, returns verification result
# ----------------------------
@app.post("/verify")
async def verify_faces(
    image1: UploadFile = File(..., description="First face image (JPG/PNG)"),
    image2: UploadFile = File(..., description="Second face image (JPG/PNG)"),
):
    """
    ## Face Verification Endpoint

    Upload two face images and get:
    - **result**: "same person" or "different person"
    - **similarity_score**: distance score (lower = more similar)
    - **verified**: boolean
    - **face1_bbox / face2_bbox**: bounding boxes [x, y, w, h]
    """

    # --- Validate file types ---
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if image1.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Image 1: unsupported type '{image1.content_type}'")
    if image2.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Image 2: unsupported type '{image2.content_type}'")

    # --- Save uploaded files temporarily ---
    uid = str(uuid.uuid4())[:8]
    ext1 = os.path.splitext(image1.filename)[-1] or ".jpg"
    ext2 = os.path.splitext(image2.filename)[-1] or ".jpg"

    path1 = os.path.join(UPLOAD_DIR, f"{uid}_face1{ext1}")
    path2 = os.path.join(UPLOAD_DIR, f"{uid}_face2{ext2}")

    try:
        with open(path1, "wb") as f:
            shutil.copyfileobj(image1.file, f)
        with open(path2, "wb") as f:
            shutil.copyfileobj(image2.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save images: {str(e)}")

    # --- Run face verification ---
    try:
        result = load_model_and_predict(path1, path2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        # Always clean up temp files
        for p in [path1, path2]:
            if os.path.exists(p):
                os.remove(p)

    # --- Handle errors from predict ---
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return JSONResponse(content=result)


# ----------------------------
# Run directly (for local testing)
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
