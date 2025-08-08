import uvicorn
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
import numpy as np
from PIL import Image
from keras.layers import TFSMLayer

app = FastAPI()

prod_model = TFSMLayer("../models/potato/1", call_endpoint="serving_default")
beta_model = TFSMLayer("../models/potato/2", call_endpoint="serving_default")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = np.array(image).astype(np.float32)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = prod_model(img_batch)
    pred_array = list(predictions.values())[0].numpy()

    predicted_class = CLASS_NAMES[np.argmax(pred_array[0])]
    confidence = np.max(pred_array[0])

    return {
        "class" : predicted_class,
        "confidence" : float(confidence),
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

