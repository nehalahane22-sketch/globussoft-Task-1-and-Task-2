# Globussoft Data Science Assignment

## Project Structure

```
globussoft-task/
├── task1_amazon_scraper.py   # Task 1 - Amazon scraper
├── task2_train.py            # Task 2 - Model setup/download
├── task2_predict.py          # Task 2 - Predict function
├── task2_main.py             # Task 2 - FastAPI app
├── requirements.txt
└── README.md
```

---

## Task 1 — Amazon Laptop Scraper

Scrapes laptop listings from amazon.in and saves to a timestamped CSV.

### Fields collected:
| Field | Description |
|-------|-------------|
| Title | Product name |
| Price | Price in ₹ |
| Rating | Star rating |
| Image URL | Product image link |
| Product URL | Link to product page |
| Ad/Organic | Whether result is sponsored or organic |

### How to run (Google Colab):
```python
!pip install requests beautifulsoup4 pandas lxml
# Upload task1_amazon_scraper.py then:
%run task1_amazon_scraper.py
```

Output: `amazon_laptops_YYYYMMDD_HHMMSS.csv`

---

## Task 2 — Face Authentication (FastAPI)

Verifies whether two face images belong to the same person using **FaceNet** via DeepFace.

### API Endpoint

**POST** `/verify`

| Parameter | Type | Description |
|-----------|------|-------------|
| image1 | File | First face image (JPG/PNG) |
| image2 | File | Second face image (JPG/PNG) |

### Response
```json
{
  "result": "same person",
  "verified": true,
  "similarity_score": 0.31,
  "threshold_used": 0.4,
  "face1_bbox": [45, 30, 120, 130],
  "face2_bbox": [50, 25, 118, 128]
}
```

### How to run (Google Colab):
```python
!pip install fastapi uvicorn deepface tf-keras python-multipart pyngrok

# Step 1: Run train.py to download model weights
%run task2_train.py

# Step 2: Start FastAPI with public URL via ngrok
from pyngrok import ngrok
import subprocess, threading

def run_server():
    subprocess.run(["uvicorn", "task2_main:app", "--host", "0.0.0.0", "--port", "8000"])

threading.Thread(target=run_server, daemon=True).start()
import time; time.sleep(3)

public_url = ngrok.connect(8000)
print("API live at:", public_url)
print("Swagger UI:", str(public_url) + "/docs")
```

Then open the Swagger UI URL to test the API interactively.

### Test with curl:
```bash
curl -X POST "{API_URL}/verify" \
  -F "image1=@face1.jpg" \
  -F "image2=@face2.jpg"
```

---

## Installation

```bash
pip install -r requirements.txt
```
