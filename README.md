
# DeepLip – Visual Speech Recognition Using Deep Learning

DeepLip is an end-to-end **lip reading (visual speech recognition)** system that converts spoken words into text using **video-only input**. The project leverages deep learning techniques to extract spatiotemporal features from lip movements and decode them into character-level text sequences.

In addition to the training and inference pipeline, the project includes a **Streamlit web application** that allows users to upload videos and obtain predictions through an intuitive UI.

---

## 🚀 Features

- End-to-end deep learning pipeline for lip reading
- Video preprocessing and lip region extraction
- Character-level sequence modeling
- CTC-based decoding for variable-length outputs
- Pretrained model support
- Streamlit web app for interactive inference
- GPU-compatible TensorFlow implementation

---

## 🧠 Model Overview

The system follows a standard visual speech recognition architecture:

1. **Video Preprocessing**
   - Frame extraction using OpenCV
   - Grayscale conversion
   - Lip-region cropping
   - Normalization across frames

2. **Feature Extraction**
   - Spatiotemporal modeling using convolutional layers
   - Temporal dependency learning using recurrent layers (LSTM/GRU)

3. **Sequence Prediction**
   - Character vocabulary encoding
   - Connectionist Temporal Classification (CTC) loss
   - Greedy/beam decoding during inference

---

## 📁 Project Structure

```

.
├── data/                 # Dataset (videos + alignment files)
├── Deeplip.ipynb         # Model training and experimentation notebook
├── app.py                # Streamlit application
├── models/               # Saved models and checkpoints
├── requirements.txt      # Python dependencies
└── README.md

````

---

### ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/deeplip.git
cd deeplip
````

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

The complete training pipeline is available in the notebook:

```text
Deeplip.ipynb
```

It covers:

* Dataset download and extraction
* Video and alignment loading
* Model architecture definition
* Training configuration
* Evaluation and prediction

Run the notebook step-by-step to train the model from scratch or adapt it for experimentation.

---

## 🎥 Making Predictions (Notebook)

You can test the model on individual video files directly from the notebook:

```python
prediction = model.predict(video)
decoded_text = decode_predictions(prediction)
```

---

## 🌐 Streamlit Web App

The Streamlit app provides a simple interface to test lip reading predictions.

### Run the App

```bash
streamlit run streamlitapp.py
```

### App Features

* Upload a video containing visible lip movements
* Automatic preprocessing and inference
* Display of predicted text output
* Clean and user-friendly interface

---

## 📦 Dependencies

Key libraries used:

* TensorFlow
* OpenCV
* NumPy
* ImageIO
* Matplotlib
* Streamlit
* gdown

Refer to `requirements.txt` for the complete list.

---

## 📊 Dataset

The project uses aligned video-text datasets where:

* Each video contains a single spoken word or phrase
* Corresponding alignment files provide ground-truth text

Dataset download is handled programmatically within the notebook.

---

## 🔮 Future Improvements

* Real-time webcam inference
* Beam search decoding
* Transformer-based temporal modeling
* Multi-word sentence prediction
* Improved lip landmark detection

---

## 👤 Author

Developed by **Mohammed Zaid V**
