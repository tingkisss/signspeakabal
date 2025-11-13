"""
model_utils.py
Helper functions untuk loading dan prediksi model CNN (versi TensorFlow Lite)
"""

import json
import numpy as np
from pathlib import Path
import tflite_runtime.interpreter as tflite

# =========================
# 📁 Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# =========================
# 🧠 Load CNN Model (TFLite)
# =========================
def load_cnn_model():
    """
    Load CNN model TFLite dan class names
    Returns: (interpreter, class_names) atau (None, None) jika gagal
    """
    try:
        model_path = MODELS_DIR / "bisindo_cnn.tflite"
        classes_path = MODELS_DIR / "class_names.json"
        
        # Check if files exist
        if not model_path.exists():
            print(f"⚠️ Model TFLite tidak ditemukan: {model_path}")
            return None, None
        
        if not classes_path.exists():
            print(f"⚠️ Class names tidak ditemukan: {classes_path}")
            return None, None
        
        print(f"📥 Loading TFLite model dari: {model_path}")
        interpreter = tflite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        # Load class names
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)
        
        print(f"✅ TFLite model loaded: {len(class_names)} classes")
        return interpreter, class_names
        
    except Exception as e:
        print(f"❌ Error loading TFLite model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# =========================
# 🔮 CNN Prediction (TFLite)
# =========================
def predict_cnn(interpreter, class_names, input_img):
    """
    Prediksi menggunakan model TFLite
    
    Args:
        interpreter: TFLite Interpreter
        class_names: List of class names
        input_img: Preprocessed image (batch_size, height, width, channels)
    
    Returns:
        (label, confidence)
    """
    if interpreter is None or class_names is None:
        raise ValueError("Model atau class names belum dimuat")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Pastikan tipe data sesuai
    input_data = input_img.astype(np.float32)
    
    # Set input tensor dan jalankan inferensi
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_idx = int(np.argmax(predictions))
    confidence = float(predictions[pred_idx])
    
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
    print("=" * 60)
    print("🧪 Testing TFLite Model Loading...")
    print("=" * 60)
    
    cnn_model, class_names = load_cnn_model()
    if cnn_model is not None:
        print(f"✅ TFLite Model: OK ({len(class_names)} classes)")
    else:
        print("❌ TFLite Model: FAILED")
    
    print("=" * 60)

if __name__ == "__main__":
    test_models()
