# ============================================================
# TASK 2 - Face Authentication | predict.py
# Purpose: Load model and verify if two face images are
#          the same person or different person
# ============================================================

# ----------------------------
# STEP 1: Install (run once in Colab)
# ----------------------------
# !pip install deepface tf-keras opencv-python-headless numpy

import os
import cv2
import numpy as np
from deepface import DeepFace

# ----------------------------
# STEP 2: Core prediction function
# This is the only function needed for testing
# ----------------------------
def load_model_and_predict(image1_path: str, image2_path: str) -> dict:
    """
    Loads FaceNet model and verifies whether two face images
    belong to the same person.

    Args:
        image1_path (str): File path to the first face image
        image2_path (str): File path to the second face image

    Returns:
        dict with keys:
            - result         : "same person" or "different person"
            - similarity_score: float (0 = identical, higher = more different)
            - verified       : bool
            - face1_bbox     : [x, y, w, h] bounding box of face in image 1
            - face2_bbox     : [x, y, w, h] bounding box of face in image 2
            - error          : error message string (only if something fails)
    """

    # --- Validate files exist ---
    if not os.path.exists(image1_path):
        return {"error": f"Image 1 not found: {image1_path}"}
    if not os.path.exists(image2_path):
        return {"error": f"Image 2 not found: {image2_path}"}

    try:
        # --- Step A: Verify the two faces using FaceNet ---
        verification = DeepFace.verify(
            img1_path   = image1_path,
            img2_path   = image2_path,
            model_name  = "Facenet",      # FaceNet model
            detector_backend = "opencv",  # face detector
            enforce_detection = False,    # don't crash if no face found
        )

        verified       = verification["verified"]          # True / False
        distance       = round(verification["distance"], 4) # similarity score

        # --- Step B: Detect faces to get bounding boxes ---
        bbox1 = get_face_bbox(image1_path)
        bbox2 = get_face_bbox(image2_path)

        return {
            "result":          "same person" if verified else "different person",
            "verified":        verified,
            "similarity_score": distance,
            "threshold_used":  round(verification.get("threshold", 0.4), 4),
            "face1_bbox":      bbox1,
            "face2_bbox":      bbox2,
        }

    except Exception as e:
        return {"error": str(e)}


# ----------------------------
# STEP 3: Helper - get face bounding box
# ----------------------------
def get_face_bbox(image_path: str) -> list:
    """
    Detects the face in an image and returns its bounding box.

    Returns:
        [x, y, width, height] or [] if no face detected
    """
    try:
        faces = DeepFace.extract_faces(
            img_path         = image_path,
            detector_backend = "opencv",
            enforce_detection = False,
        )
        if faces:
            region = faces[0].get("facial_area", {})
            return [
                region.get("x", 0),
                region.get("y", 0),
                region.get("w", 0),
                region.get("h", 0),
            ]
    except Exception:
        pass
    return []


# ----------------------------
# STEP 4: Test it (run directly)
# ----------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  Face Authentication - Predict Script")
    print("=" * 55)

    # --- Option A: Test with your own images ---
    # Change these paths to your actual image files
    IMG1 = "sample_images/person1_a.jpg"
    IMG2 = "sample_images/person1_b.jpg"

    # --- Option B (Colab): Upload images interactively ---
    try:
        from google.colab import files
        print("\n📁 Upload Image 1 (first face):")
        uploaded1 = files.upload()
        IMG1 = list(uploaded1.keys())[0]

        print("\n📁 Upload Image 2 (second face):")
        uploaded2 = files.upload()
        IMG2 = list(uploaded2.keys())[0]
    except ImportError:
        pass  # Not in Colab, use paths above

    # --- Run prediction ---
    print(f"\n🔍 Comparing:\n  Image 1: {IMG1}\n  Image 2: {IMG2}\n")
    result = load_model_and_predict(IMG1, IMG2)

    # --- Display result ---
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Result          : {result['result'].upper()}")
        print(f"   Verified        : {result['verified']}")
        print(f"   Similarity Score: {result['similarity_score']}")
        print(f"   Threshold Used  : {result['threshold_used']}")
        print(f"   Face 1 BBox     : {result['face1_bbox']}")
        print(f"   Face 2 BBox     : {result['face2_bbox']}")
