import os
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
from PIL import Image

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
TFSERVER_URL = os.getenv("TFSERVER_URL", "http://tfserving:8501")  # docker‑compose service name
MODEL_NAME   = "potato"
DEFAULT_VER  = None   # e.g. 1 or 2 if you want too hard‑pin; leave None to use "latest"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# ---------------------------------------------------------------------------
app = FastAPI(title="Potato Disease Classifier")

@app.get("/ping")
async def ping():
    return {"status": "alive"}

# ---------------------------------------------------------------------------
# 2. UTILITIES
# ---------------------------------------------------------------------------
def read_file_as_image(data: bytes) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert("RGB")
    img = np.array(img).astype(np.float32)           # (H, W, 3)
    return img

def tf_serving_predict(instances: np.ndarray, version: int | None = None):
    """
    POSTs the given batch to TF‑Serving and returns its JSON-decoded response.
    """
    if version:
        url = f"{TFSERVER_URL}/v1/models/{MODEL_NAME}/versions/{version}:predict"
    else:
        url = f"{TFSERVER_URL}/v1/models/{MODEL_NAME}:predict"

    payload = {"instances": instances.tolist()}      # lists/JSON – not NumPy
    response = requests.post(url, json=payload, timeout=10)

    if response.status_code != 200:
        raise RuntimeError(f"TF‑Serving error {response.status_code}: {response.text}")

    return response.json()

# ---------------------------------------------------------------------------
# 3. ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        version: int | None = DEFAULT_VER         # optional query param ?version=2
):
    image = read_file_as_image(await file.read())
    batch = np.expand_dims(image, 0)          # (1, H, W, 3)

    try:
        result = tf_serving_predict(batch, version)
        # TF‑Serving returns {"predictions":[[..]]} or similar
        preds = np.array(result["predictions"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    predicted_class = CLASS_NAMES[int(np.argmax(preds[0]))]
    confidence      = float(np.max(preds[0]))

    return {
        "version": version if version else "latest",
        "class": predicted_class,
        "confidence": confidence
    }

# ---------------------------------------------------------------------------
# Optional local dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
