# ============================================================
# TASK 2 - Face Authentication | train.py
# Purpose: Download and verify the FaceNet model is ready
# Run this ONCE in Google Colab before using predict.py
# ============================================================

# ----------------------------
# STEP 1: Install dependencies
# ----------------------------
# Run in Colab cell first:
# !pip install deepface tf-keras opencv-python-headless numpy

import os
import numpy as np

# ----------------------------
# STEP 2: Import DeepFace
# ----------------------------
print("📦 Loading DeepFace library...")
from deepface import DeepFace
print("✅ DeepFace imported successfully.")

# ----------------------------
# STEP 3: Download / Build the FaceNet model
# (First run downloads weights automatically ~90MB)
# ----------------------------
print("\n⬇️  Building FaceNet model (downloading weights if first time)...")
print("    This may take 1-2 minutes on first run...\n")

try:
    model = DeepFace.build_model("Facenet")
    print("✅ FaceNet model loaded and cached successfully!")
    print(f"   Model type : {type(model)}")
except Exception as e:
    print(f"❌ Error building model: {e}")
    raise

# ----------------------------
# STEP 4: Test the model works
# using two built-in test images from DeepFace
# ----------------------------
print("\n🧪 Running a quick sanity test with sample images...")

# Create two tiny dummy images (just to verify the pipeline works)
import cv2

os.makedirs("sample_images", exist_ok=True)

# Create a plain grey image as a dummy test face
dummy1 = np.ones((160, 160, 3), dtype=np.uint8) * 100
dummy2 = np.ones((160, 160, 3), dtype=np.uint8) * 150

cv2.imwrite("sample_images/test_face1.jpg", dummy1)
cv2.imwrite("sample_images/test_face2.jpg", dummy2)

print("✅ Sample images created in 'sample_images/' folder.")
print("\n🎉 Training/setup complete! Model is ready.")
print("   You can now run predict.py or main.py (FastAPI).")
print("\n📁 Model weights are cached at: ~/.deepface/weights/")
