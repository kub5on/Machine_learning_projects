## Plant Seedlings Classifier 🌻

This project classifies plant seedlings photos using an **MobileNetV2** model. The system takes images of seedlings, extracts features, and predicts the plant type. It includes a **FastAPI** backend for predictions and a **Streamlit** frontend for easy interaction.

### Features
- **MobileNetV2 Model:** Pretrained encoder with custom Dense classification head.  
- **Data Augmentation:** Rotation, zoom, horizontal flip, and rescaling.  
- **Prediction API:** FastAPI endpoint for image-based predictions.  
- **Web Interface:** Streamlit frontend for uploading images and viewing results.  
- **Confidence Scores:** Provides probability for each predicted class.  
- **Reproducibility:** Deterministic setup with fixed seeds for Python, NumPy, and TensorFlow.
- **Pretrained model:** `plant_seedlings_model.h5`

### Technologies
`Python 3.10+`, `TensorFlow`, `Keras`, `FastAPI`, `Streamlit`, `NumPy`, `Pandas`, `Pillow`, `Requests`

### ⚡ Quick Start
1. Add `plants_train/` and `plants_test/` folders with images to the project folder.
2. Install dependencies: ```pip install -r requirements.txt```.
3. Run: ```python main.py``` to train the model.
4. Run: ```uvicorn app:app --reload``` to start the backend API.
5. Run: ```streamlit run app_frontend.py``` to launch the frontend.
6. Upload different seedling images in Streamlit and view predictions.

### Possible classes
Black-grass, Charlock, Cleavers, Common Chickweed, Common wheat, Fat Hen, Loose Silky-bent, Maize, Scentless Mayweed, Shepherd's Purse, Small-flowered Cranesbill, Sugar beet





