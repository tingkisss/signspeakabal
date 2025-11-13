import os
import time
import json
import base64
import traceback
import tempfile
import threading
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2

# Integrasi model_utils
from model_utils import load_cnn_model, predict_cnn

# =========================
# 📁 Path & Konfigurasi
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = BASE_DIR / "temp_videos"
LOGS_DIR = BASE_DIR / "logs"

LOG_FILE = LOGS_DIR / "server.log"

IMG_HEIGHT = 128
IMG_WIDTH = 128
ENABLE_AUTO_RELOAD = os.environ.get("AUTO_RELOAD_MODELS", "0") == "1"
MODEL_POLL_INTERVAL = 10

# =========================
# 🧠 Global Models
# =========================
CNN_MODEL = None
CLASS_NAMES = None
_models_lock = threading.Lock()
_model_load_time = 0

# =========================
# ⚙️ Utilities
# =========================
def ensure_dirs():
    for d in [MODELS_DIR, DATA_DIR, VIDEOS_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

def log_event(msg):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# =========================
# 🔄 Load Models
# =========================
def load_all_models():
    global CNN_MODEL, CLASS_NAMES, _model_load_time
    with _models_lock:
        log_event("🔄 Memuat CNN model...")
        try:
            CNN_MODEL, CLASS_NAMES = load_cnn_model()
            _model_load_time = time.time()
            log_event(f"✅ CNN={'Loaded' if CNN_MODEL else 'None'}")
        except Exception as e:
            log_event(f"❌ Error loading model: {e}")
            traceback.print_exc()

# =========================
# 👁️ Poller (optional)
# =========================
def model_poller(stop_event):
    if not ENABLE_AUTO_RELOAD:
        log_event("ℹ️ Auto-reload nonaktif. Gunakan /reload_models untuk manual reload.")
        return
    
    last_time = 0
    while not stop_event.is_set():
        time.sleep(MODEL_POLL_INTERVAL)
        try:
            model_file = MODELS_DIR / "bisindo_cnn.keras"
            
            if not model_file.exists():
                continue
                
            cur_time = os.path.getmtime(model_file)
            
            if last_time == 0:
                last_time = cur_time
            elif cur_time > last_time:
                log_event("👁️ Model file berubah → reload otomatis...")
                load_all_models()
                last_time = cur_time
        except Exception as e:
            log_event(f"⚠️ Poller error: {e}")
            continue

# =========================
# 🧩 Helper
# =========================
def safe_b64_to_bytes(data_b64: str) -> bytes:
    """Convert base64 string to bytes with proper padding"""
    if data_b64.startswith("data:"):
        parts = data_b64.split(",", 1)
        if len(parts) == 2:
            data_b64 = parts[1]
    
    # Add padding if needed
    padding = len(data_b64) % 4
    if padding != 0:
        data_b64 += "=" * (4 - padding)
    
    try:
        return base64.b64decode(data_b64)
    except Exception as e:
        log_event(f"❌ Base64 decode error: {e}")
        raise ValueError("Invalid base64 data")

def preprocess_bgr_image(bgr_img):
    """Preprocess BGR image for model input"""
    if bgr_img is None or bgr_img.size == 0:
        raise ValueError("Invalid image")
    
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_WIDTH, IMG_HEIGHT))
    arr = resized.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

# =========================
# 🚀 Flask App
# =========================
app = Flask(__name__)
CORS(app)

# Error handler
@app.errorhandler(Exception)
def handle_error(e):
    log_event(f"❌ Unhandled error: {e}")
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    classes = []
    if CLASS_NAMES is not None:
        classes = [str(c) for c in CLASS_NAMES]
    
    return jsonify({
        "status": "ok",
        "cnn_loaded": bool(CNN_MODEL is not None),
        "classes": classes,
        "version": "5.0 CNN-Only",
        "uptime": float(time.time() - _model_load_time) if _model_load_time else 0
    })

# =========================
# 🧠 CNN Predict (Image)
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if CNN_MODEL is None:
            log_event("⚠️ /predict called but CNN not loaded")
            return jsonify({"error": "Model CNN belum dimuat"}), 503
        
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "Gambar tidak ditemukan"}), 400
        
        # Decode image
        raw = safe_b64_to_bytes(data["image"])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Gagal decode gambar"}), 400
        
        # Preprocess and predict
        inp = preprocess_bgr_image(img)
        label, conf = predict_cnn(CNN_MODEL, CLASS_NAMES, inp)
        
        log_event(f"🔮 /predict => {label} ({conf:.4f})")
        return jsonify({"label": label, "confidence": float(conf)})
        
    except ValueError as ve:
        log_event(f"⚠️ /predict validation error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        log_event(f"❌ /predict error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =========================
# 🎥 CNN Predict (Video)
# =========================
@app.route("/predict_video", methods=["POST"])
def predict_video():
    temp_path = None
    try:
        if CNN_MODEL is None:
            return jsonify({"error": "Model CNN belum dimuat"}), 503
        
        data = request.json
        if not data or "video" not in data:
            return jsonify({"error": "Video tidak ditemukan"}), 400
        
        raw = safe_b64_to_bytes(data["video"])
        ensure_dirs()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm", dir=str(VIDEOS_DIR)) as tmp:
            tmp.write(raw)
            temp_path = tmp.name
        
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify({"error": "Tidak bisa membuka video"}), 400
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return jsonify({"error": "Video kosong"}), 400
        
        # Sample max 30 frames
        indices = np.linspace(0, total_frames - 1, min(total_frames, 30), dtype=int)
        preds = []
        
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret:
                continue
            
            try:
                inp = preprocess_bgr_image(frame)
                pred = CNN_MODEL.predict(inp, verbose=0)[0]
                preds.append(pred)
            except Exception:
                continue
        
        cap.release()
        
        if not preds:
            return jsonify({"error": "Tidak ada frame valid"}), 400
        
        # Average predictions
        avg = np.mean(preds, axis=0)
        idx = int(np.argmax(avg))
        label = CLASS_NAMES[idx] if CLASS_NAMES and idx < len(CLASS_NAMES) else f"class_{idx}"
        conf = float(avg[idx])
        
        log_event(f"🎥 /predict_video => {label} ({conf:.4f}) from {len(preds)} frames")
        return jsonify({
            "label": label,
            "confidence": conf,
            "frames_analyzed": len(preds)
        })
        
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        log_event(f"❌ /predict_video error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

# =========================
# 🔄 Reload Models
# =========================
@app.route("/reload_models", methods=["POST"])
def reload_models():
    try:
        load_all_models()
        return jsonify({
            "message": "Model berhasil dimuat ulang",
            "cnn_loaded": CNN_MODEL is not None
        })
    except Exception as e:
        log_event(f"❌ /reload_models error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =========================
# ❤️ Health Check
# =========================
@app.route("/health", methods=["GET"])
def health():
    cnn_classes = []
    if CLASS_NAMES is not None:
        cnn_classes = [str(c) for c in CLASS_NAMES]
    
    return jsonify({
        "status": "healthy",
        "cnn_ready": bool(CNN_MODEL is not None),
        "cnn_classes": cnn_classes,
        "uptime_seconds": float(time.time() - _model_load_time) if _model_load_time else 0
    })

# =========================
# 🎯 Main
# =========================
if __name__ == "__main__":
    ensure_dirs()
    log_event("🚀 SignSpeak Backend v5.0 (CNN-Only) starting...")
    
    # Load models on startup
    load_all_models()
    
    # Start poller thread
    stop_event = threading.Event()
    poller = threading.Thread(target=model_poller, args=(stop_event,), daemon=True)
    poller.start()
    
    try:
        # Run Flask
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        log_event("⚠️ Keyboard interrupt received")
    finally:
        stop_event.set()
        poller.join(timeout=2)
        log_event("🛑 Server stopped.")