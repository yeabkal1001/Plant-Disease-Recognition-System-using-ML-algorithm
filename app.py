from flask import Flask, render_template, request, redirect, send_from_directory, jsonify
import numpy as np
import json
import uuid
import pickle
import os
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

IMG_SIZE = (128, 128)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_1 = None
model_2 = None
scaler = None
comparison_data = None

try:
    model_path_1 = "models/algorithm_1_rf.pkl"
    if os.path.exists(model_path_1):
        with open(model_path_1, "rb") as f:
            model_1 = pickle.load(f)
        logger.info("✓ Loaded Algorithm 1 (Random Forest)")
    else:
        logger.warning("Model 1 not found at " + model_path_1)
except Exception as e:
    logger.error(f"Failed to load Algorithm 1: {e}")

try:
    model_path_2 = "models/algorithm_2_gb.pkl"
    if os.path.exists(model_path_2):
        with open(model_path_2, "rb") as f:
            model_2 = pickle.load(f)
        logger.info("✓ Loaded Algorithm 2 (Gradient Boosting)")
    else:
        logger.warning("Model 2 not found at " + model_path_2)
except Exception as e:
    logger.error(f"Failed to load Algorithm 2: {e}")

try:
    scaler_path = "models/feature_scaler.pkl"
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info("✓ Loaded feature scaler")
    else:
        logger.warning("Scaler not found at " + scaler_path)
except Exception as e:
    logger.error(f"Failed to load scaler: {e}")

try:
    comparison_path = "models/comparison.json"
    if os.path.exists(comparison_path):
        with open(comparison_path, "r") as f:
            comparison_data = json.load(f)
        logger.info("✓ Loaded comparison data")
except Exception as e:
    logger.error(f"Failed to load comparison data: {e}")

# Load class names dynamically if available, otherwise use default
class_names_path = "models/class_names.json"
if os.path.exists(class_names_path):
    with open(class_names_path, "r") as f:
        class_data = json.load(f)
        label = class_data.get("class_names", [])
    print(f"[OK] Loaded {len(label)} class names from training data")
else:
    # Default labels (fallback)
    label = ['Apple___Apple_scab',
     'Apple___Black_rot',
     'Apple___Cedar_apple_rust',
     'Apple___healthy',
     'Background_without_leaves',
     'Blueberry___healthy',
     'Cherry___Powdery_mildew',
     'Cherry___healthy',
     'Corn___Cercospora_leaf_spot Gray_leaf_spot',
     'Corn___Common_rust',
     'Corn___Northern_Leaf_Blight',
     'Corn___healthy',
     'Grape___Black_rot',
     'Grape___Esca_(Black_Measles)',
     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
     'Grape___healthy',
     'Orange___Haunglongbing_(Citrus_greening)',
     'Peach___Bacterial_spot',
     'Peach___healthy',
     'Pepper,_bell___Bacterial_spot',
     'Pepper,_bell___healthy',
     'Potato___Early_blight',
     'Potato___Late_blight',
     'Potato___healthy',
     'Raspberry___healthy',
     'Soybean___healthy',
     'Squash___Powdery_mildew',
     'Strawberry___Leaf_scorch',
     'Strawberry___healthy',
     'Tomato___Bacterial_spot',
     'Tomato___Early_blight',
     'Tomato___Late_blight',
     'Tomato___Leaf_Mold',
     'Tomato___Septoria_leaf_spot',
     'Tomato___Spider_mites Two-spotted_spider_mite',
     'Tomato___Target_Spot',
     'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
     'Tomato___Tomato_mosaic_virus',
     'Tomato___healthy']
    print(f"[OK] Using default {len(label)} class labels")

with open("plant_disease.json", 'r') as file:
    disease_list = json.load(file)
    plant_disease = {disease['name']: disease for disease in disease_list}

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

def extract_features_from_image(img_array):
    features = []
    
    if len(img_array.shape) == 3:
        gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    else:
        gray = img_array
    
    gray = gray.astype(np.uint8)
    
    flat_features = gray.flatten() / 255.0
    features.extend(flat_features[:100])
    
    features.append(np.mean(gray))
    features.append(np.std(gray))
    features.append(np.var(gray))
    features.append(np.max(gray) - np.min(gray))
    
    h, w = gray.shape
    mid_h, mid_w = h // 2, w // 2
    center = gray[mid_h-32:mid_h+32, mid_w-32:mid_w+32] # Increased center region to match training
    if center.size > 0:
        features.append(np.mean(center))
        features.append(np.std(center))
    else:
        features.extend([0, 0])
    
    edges_h = np.abs(np.diff(gray, axis=0)).sum()
    edges_v = np.abs(np.diff(gray, axis=1)).sum()
    features.append(edges_h)
    features.append(edges_v)
    
    if len(img_array.shape) == 3:
        for i in range(3):
            channel = img_array[:, :, i]
            features.append(np.mean(channel))
            features.append(np.std(channel))
            features.append(np.max(channel))
            features.append(np.min(channel))
            # Added more color features to match training
            features.append(np.median(channel))
    else:
        # If grayscale, duplicate features
        features.extend([np.mean(gray), np.std(gray), np.max(gray), np.min(gray), np.median(gray)] * 3)
    
    features_array = np.array(features, dtype=np.float32)
    
    # Pad to 200 features to match training
    if len(features_array) < 200:
        padding = np.zeros(200 - len(features_array), dtype=np.float32)
        features_array = np.concatenate([features_array, padding])
    
    return np.array([features_array[:200]], dtype=np.float32)

def model_predict(image):
    try:
        img = Image.open(image).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.uint8)
        
        features = extract_features_from_image(img_array)
        
        if scaler:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        result = {}
        best_confidence = 0
        best_result = None
        best_model = None
        CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to accept prediction (30%)
        
        if model_1:
            prediction_1 = model_1.predict(features_scaled)[0]
            proba_1 = model_1.predict_proba(features_scaled)
            conf_1 = float(np.max(proba_1))
            label_1 = label[prediction_1]
            
            # Check if prediction is background or low confidence
            if label_1 == "Background_without_leaves" or conf_1 < CONFIDENCE_THRESHOLD:
                disease_1 = {
                    "name": "Not a Plant Leaf",
                    "cause": "The uploaded image does not appear to be a plant leaf. Please upload an image of a plant leaf for disease detection.",
                    "cure": "Please ensure the image shows a clear view of a plant leaf."
                }
            else:
                disease_1 = plant_disease.get(label_1, {"name": label_1, "cause": "Unknown", "cure": "Unknown"})
            
            result["model_1"] = {
                "name": "Algorithm 1 (Random Forest)",
                "prediction": disease_1,
                "confidence": f"{conf_1*100:.2f}%"
            }
            
            if conf_1 > best_confidence:
                best_confidence = conf_1
                best_result = disease_1
                best_model = "model_1"
        
        if model_2:
            prediction_2 = model_2.predict(features_scaled)[0]
            proba_2 = model_2.predict_proba(features_scaled)
            conf_2 = float(np.max(proba_2))
            label_2 = label[prediction_2]
            
            # Check if prediction is background or low confidence
            if label_2 == "Background_without_leaves" or conf_2 < CONFIDENCE_THRESHOLD:
                disease_2 = {
                    "name": "Not a Plant Leaf",
                    "cause": "The uploaded image does not appear to be a plant leaf. Please upload an image of a plant leaf for disease detection.",
                    "cure": "Please ensure the image shows a clear view of a plant leaf."
                }
            else:
                disease_2 = plant_disease.get(label_2, {"name": label_2, "cause": "Unknown", "cure": "Unknown"})
            
            result["model_2"] = {
                "name": "Algorithm 2 (Gradient Boosting)",
                "prediction": disease_2,
                "confidence": f"{conf_2*100:.2f}%"
            }
            
            if conf_2 > best_confidence:
                best_confidence = conf_2
                best_result = disease_2
                best_model = "model_2"
        
        # Final check: if best confidence is still too low, override result
        if best_confidence < CONFIDENCE_THRESHOLD and best_result and best_result.get("name") != "Not a Plant Leaf":
            best_result = {
                "name": "Low Confidence - May Not Be a Leaf",
                "cause": f"The system detected '{best_result.get('name', 'unknown')}' with only {best_confidence*100:.1f}% confidence. The image may not be a plant leaf. Please upload a clear image of a plant leaf.",
                "cure": "Please ensure you upload a clear, well-lit image of a plant leaf for accurate disease detection."
            }
        
        if best_result:
            result["best"] = best_result
            result["best_model"] = best_model
            result["confidence_warning"] = best_confidence < CONFIDENCE_THRESHOLD
        
        if comparison_data:
            # Handle both old format (accuracy) and new format (test_accuracy)
            algo1_data = comparison_data.get("algorithm_1_random_forest", {})
            algo2_data = comparison_data.get("algorithm_2_gradient_boosting", {})
            
            model_1_acc = algo1_data.get("test_accuracy") or algo1_data.get("accuracy", 0)
            model_2_acc = algo2_data.get("test_accuracy") or algo2_data.get("accuracy", 0)
            
            result["comparison_info"] = {
                "model_1_name": algo1_data.get("name", "Random Forest"),
                "model_1_train_accuracy": algo1_data.get("train_accuracy", 0),
                "model_1_test_accuracy": model_1_acc,
                "model_2_name": algo2_data.get("name", "Gradient Boosting"),
                "model_2_train_accuracy": algo2_data.get("train_accuracy", 0),
                "model_2_test_accuracy": model_2_acc,
                "best_model_overall": comparison_data.get("best_model", ""),
                "best_model_name": comparison_data.get("best_model_name", "")
            }
        
        print(f"[DEBUG] Result keys: {result.keys()}")
        return result
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": str(e)}

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        try:
            if 'img' not in request.files:
                error_msg = "No image file provided"
                logger.warning(error_msg)
                return render_template('home.html', error=error_msg), 400
            
            image = request.files['img']
            
            if image.filename == '':
                error_msg = "No file selected"
                logger.warning(error_msg)
                return render_template('home.html', error=error_msg), 400
            
            if not allowed_file(image.filename):
                error_msg = "File type not allowed. Please upload PNG, JPG, JPEG, GIF, or BMP images."
                logger.warning(f"Invalid file type: {image.filename}")
                return render_template('home.html', error=error_msg), 400
            
            # Ensure uploadimages directory exists
            os.makedirs('uploadimages', exist_ok=True)
            
            temp_id = uuid.uuid4().hex
            image_filename = f'temp_{temp_id}_{image.filename}'
            full_path = f'./uploadimages/{image_filename}'
            
            image.save(full_path)
            logger.info(f"Image saved: {image_filename}")
            
            if not (model_1 or model_2):
                error_msg = "No trained models available. Please train models first."
                logger.error(error_msg)
                return render_template('home.html', error=error_msg), 503
            
            prediction_data = model_predict(full_path)
            logger.info(f"Prediction completed for {image_filename}")
            
            if "error" in prediction_data:
                return render_template('home.html', error=prediction_data["error"]), 400
            
            return render_template('home.html', 
                                 result=True, 
                                 imagepath=f'/uploadimages/{image_filename}', 
                                 prediction=prediction_data.get("best"),
                                 comparison=prediction_data)
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return render_template('home.html', error=error_msg), 500
    
    else:
        return redirect('/')

if __name__ == "__main__":
    logger.info("\n" + "="*60)
    logger.info("PLANT DISEASE RECOGNITION SYSTEM")
    logger.info("="*60)
    if model_1 or model_2:
        logger.info("✓ Models loaded successfully")
        logger.info("✓ Starting Flask server at http://localhost:5000")
    else:
        logger.warning("⚠ No trained models found. Please run train_with_real_dataset.py first.")
    logger.info("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
