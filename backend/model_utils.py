"""
model_utils.py
Helper functions untuk loading dan prediksi model CNN
"""

import json
import numpy as np
from pathlib import Path
from tensorflow import keras

# =========================
# 📁 Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# =========================
# 🧠 Load CNN Model
# =========================
def load_cnn_model():
    """
    Load CNN model dan class names
    Returns: (model, class_names) atau (None, None) jika gagal
    """
    try:
        model_path = MODELS_DIR / "bisindo_cnn.keras"
        classes_path = MODELS_DIR / "class_names.json"
        
        # Check if files exist
        if not model_path.exists():
            print(f"⚠️ Model CNN tidak ditemukan: {model_path}")
            return None, None
        
        if not classes_path.exists():
            print(f"⚠️ Class names tidak ditemukan: {classes_path}")
            return None, None
        
        # Load model
        print(f"📥 Loading CNN model dari: {model_path}")
        model = keras.models.load_model(model_path)
        
        # Load class names
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
        
        print(f"✅ CNN model loaded: {len(class_names)} classes")
        return model, class_names
        
    except Exception as e:
        print(f"❌ Error loading CNN model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# =========================
# 🔮 CNN Prediction
# =========================
def predict_cnn(model, class_names, input_img):
    """
    Prediksi menggunakan CNN
    
    Args:
        model: Keras model
        class_names: List of class names
        input_img: Preprocessed image (batch_size, height, width, channels)
    
    Returns:
        (label, confidence)
    """
    if model is None or class_names is None:
        raise ValueError("Model atau class names belum dimuat")
    
    # Predict
    predictions = model.predict(input_img, verbose=0)
    pred_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][pred_idx])
    
    # Get label
    if pred_idx < len(class_names):
        label = class_names[pred_idx]
    else:
        label = f"class_{pred_idx}"
    
    return label, confidence

# =========================
# 🧪 Test Functions
# =========================
def test_models():
    """Test apakah model bisa dimuat"""
    print("="*60)
    print("🧪 Testing Model Loading...")
    print("="*60)
    
    # Test CNN
    cnn_model, class_names = load_cnn_model()
    if cnn_model is not None:
        print(f"✅ CNN Model: OK ({len(class_names)} classes)")
    else:
        print("❌ CNN Model: FAILED")
    
    print("="*60)

if __name__ == "__main__":
    # Run test if executed directly
    test_models()