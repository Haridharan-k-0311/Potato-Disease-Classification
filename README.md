# 🥔 Potato Disease Classification using CNN

A deep learning-based web application that classifies potato leaf diseases into three categories — **Early Blight**, **Late Blight**, and **Healthy** — using a Convolutional Neural Network (CNN). Built using Python, TensorFlow/Keras, and served via a Flask API.

---

## 📁 Project Structure


```markdown

Potato-Disease-Classification/
├── Dataset\_/                      # Contains raw training images
│   ├── Potato\_\_\_Early\_blight/
│   ├── Potato\_\_\_Late\_blight/
│   └── Potato\_\_\_healthy/
│
├── api/                           # Flask API to serve predictions
│   ├── main.py
│   └── requirements.txt
│
├── models/
│   └── potato/
│       ├── 1/
│       ├── 2/
│       ├── 3/
│       ├── 4/
│       └── 5/
│           ├── saved\_model.pb
│           └── variables/
│
├── training/                      # Scripts and notebooks for training
│   ├── temp.py
│   └── training.ipynb
│
├── .gitignore
└── README.md

```
---

## 🧠 Model Details

- **Model**: CNN built using TensorFlow and Keras  
- **Input Size**: 224x224 pixels  
- **Dataset**: [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- **Classes**:
  - `Potato___Early_blight`
  - `Potato___Late_blight`
  - `Potato___healthy`

---

## 🚀 How to Run (Locally)

### 1. Clone the Repository

```bash
git clone https://github.com/Haridharan-k-0311/Potato-Disease-Classification.git
cd Potato-Disease-Classification/api
````

### 2. Create Virtual Environment (optional)

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Flask App

```bash
python main.py
```

### 5. Access Web App

Go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🖼️ Example Input

Upload an image of a potato leaf — the model will return the predicted class (e.g., **Late Blight**) with confidence score.

---

## 🧪 Training Info

* Training done using `training/training.ipynb` and `temp.py`
* Used `ImageDataGenerator` with 80-20 train-validation split
* Trained model saved in `models/potato/` using TensorFlow `SavedModel` format

---

## 📦 Requirements

<details>
<summary>Click to expand</summary>

```
Flask
tensorflow
keras
numpy
opencv-python
matplotlib
Pillow
```

</details>

> Or install all at once:
>
> ```bash
> pip install -r api/requirements.txt
> ```

---

## 🙌 Author

**Haridharan K**
GitHub: [@Haridharan-k-0311](https://github.com/Haridharan-k-0311)

