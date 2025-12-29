import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import pickle
import json
import warnings
from PIL import Image
from tqdm import tqdm
import glob
import sys
import argparse
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
IMG_SIZE = (128, 128)
RANDOM_STATE = 42

# Use command line argument or environment variable, with fallback default
DATASET_PATH = os.environ.get('DATASET_PATH', 'Plant_leave_diseases_dataset_with_augmentation')

MAX_SAMPLES_PER_CLASS = 2000  # Limit samples per class for faster training
TRAIN_SIZE = 0.8  # 80% of data for training
TEST_SIZE = 0.2   # 20% of data for testing

def extract_features_from_image(img_array):
    """Extract features from image array"""
    features = []
    
    if len(img_array.shape) == 3:
        gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    else:
        gray = img_array
    
    gray = gray.astype(np.uint8)
    
    # Flattened pixel values (first 100 pixels)
    flat_features = gray.flatten() / 255.0
    features.extend(flat_features[:100])
    
    # Statistical features
    features.append(np.mean(gray))
    features.append(np.std(gray))
    features.append(np.var(gray))
    features.append(np.max(gray) - np.min(gray))
    
    # Center region features
    h, w = gray.shape
    mid_h, mid_w = h // 2, w // 2
    center = gray[mid_h-32:mid_h+32, mid_w-32:mid_w+32] # Increased center region
    if center.size > 0:
        features.append(np.mean(center))
        features.append(np.std(center))
    else:
        features.extend([0, 0])
    
    # Edge detection features
    edges_h = np.abs(np.diff(gray, axis=0)).sum()
    edges_v = np.abs(np.diff(gray, axis=1)).sum()
    features.append(edges_h)
    features.append(edges_v)
    
    # Color channel features
    if len(img_array.shape) == 3:
        for i in range(3):
            channel = img_array[:, :, i]
            features.append(np.mean(channel))
            features.append(np.std(channel))
            features.append(np.max(channel))
            features.append(np.min(channel))
            # Added more color features
            features.append(np.median(channel))
    else:
        # If grayscale, duplicate features
        features.extend([np.mean(gray), np.std(gray), np.max(gray), np.min(gray), np.median(gray)] * 3)
    
    features_array = np.array(features, dtype=np.float32)
    
    # Pad to 200 features if needed (increased from 150)
    if len(features_array) < 200:
        padding = np.zeros(200 - len(features_array), dtype=np.float32)
        features_array = np.concatenate([features_array, padding])
    
    return features_array[:200]

def load_dataset_from_folder(dataset_path):
    """Load images from the dataset folder structure"""
    logger.info("="*60)
    logger.info("LOADING DATASET FROM REAL IMAGES")
    logger.info("="*60)
    
    X = []
    y = []
    class_names = []
    class_to_idx = {}
    
    # Get all class folders
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path not found: {dataset_path}")
        logger.error("Please provide the correct dataset path.")
        return None, None, None, None
    
    class_folders = [d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))]
    class_folders.sort()
    
    logger.info(f"Found {len(class_folders)} classes")
    
    # Map class names to indices
    for idx, class_name in enumerate(class_folders):
        class_to_idx[class_name] = idx
        class_names.append(class_name)
    
    # Load images from each class
    total_images = 0
    for class_name in tqdm(class_folders, desc="Loading classes"):
        class_path = os.path.join(dataset_path, class_name)
        class_idx = class_to_idx[class_name]
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(class_path, ext)))
        
        # Limit samples per class
        if len(image_files) > MAX_SAMPLES_PER_CLASS:
            import random
            random.seed(RANDOM_STATE)
            image_files = random.sample(image_files, MAX_SAMPLES_PER_CLASS)
        
        print(f"  {class_name}: {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc=f"  Processing {class_name}", leave=False):
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(IMG_SIZE)
                img_array = np.array(img, dtype=np.uint8)
                
                features = extract_features_from_image(img_array)
                X.append(features)
                y.append(class_idx)
                total_images += 1
            except Exception as e:
                print(f"    Error loading {img_path}: {e}")
                continue
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    print(f"\nTotal images loaded: {total_images}")
    print(f"Total classes: {len(class_names)}")
    print(f"Feature vector size: {X.shape[1]}")
    
    # Save class names mapping
    with open("models/class_names.json", "w") as f:
        json.dump({"class_names": class_names, "class_to_idx": class_to_idx}, f, indent=2)
    
    return X, y, class_names, class_to_idx

def build_algorithm_1_random_forest():
    """Algorithm 1: Random Forest Classifier"""
    print("\nBuilding Algorithm 1: Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=300, # Increased from 200
        max_depth=50, # Increased from 40
        random_state=RANDOM_STATE, 
        min_samples_split=2, 
        max_features='sqrt', 
        n_jobs=-1, 
        verbose=0
    )
    return model

def build_algorithm_2_gradient_boosting():
    """Algorithm 2: Gradient Boosting Classifier"""
    print("\nBuilding Algorithm 2: Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(
        n_estimators=150, # Reduced to speed up full-data training
        learning_rate=0.1, # Increased from 0.08
        max_depth=8, # Reduced depth to decrease training time
        subsample=0.8, 
        random_state=RANDOM_STATE, 
        verbose=0
    )
    return model

def train_models(X_train, X_test, y_train, y_test, train_algo1=True, train_algo2=True):
    """Train the selected algorithms and compare performance"""
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    algo1_model = None
    acc1_train = None
    acc1_test = None
    if train_algo1:
        print("\n" + "="*60)
        print("TRAINING ALGORITHM 1: RANDOM FOREST")
        print("="*60)
        algo1_model = build_algorithm_1_random_forest()
        print("Training Random Forest...")
        algo1_model.fit(X_train_scaled, y_train)
        acc1_train = algo1_model.score(X_train_scaled, y_train)
        acc1_test = algo1_model.score(X_test_scaled, y_test)
        print(f"Algorithm 1 (Random Forest) - Train Accuracy: {acc1_train*100:.2f}%")
        print(f"Algorithm 1 (Random Forest) - Test Accuracy: {acc1_test*100:.2f}%")
        y_pred1 = algo1_model.predict(X_test_scaled)
        print("\nClassification Report for Random Forest:")
        print(classification_report(y_test, y_pred1, zero_division=0))

    algo2_model = None
    acc2_train = None
    acc2_test = None
    if train_algo2:
        print("\n" + "="*60)
        print("TRAINING ALGORITHM 2: GRADIENT BOOSTING")
        print("="*60)
        algo2_model = build_algorithm_2_gradient_boosting()
        print("Training Gradient Boosting...")
        algo2_model.fit(X_train_scaled, y_train)
        acc2_train = algo2_model.score(X_train_scaled, y_train)
        acc2_test = algo2_model.score(X_test_scaled, y_test)
        print(f"Algorithm 2 (Gradient Boosting) - Train Accuracy: {acc2_train*100:.2f}%")
        print(f"Algorithm 2 (Gradient Boosting) - Test Accuracy: {acc2_test*100:.2f}%")
        y_pred2 = algo2_model.predict(X_test_scaled)
        print("\nClassification Report for Gradient Boosting:")
        print(classification_report(y_test, y_pred2, zero_division=0))

    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    if acc1_test is not None:
        print(f"Algorithm 1 (Random Forest):      Train: {acc1_train*100:.2f}% | Test: {acc1_test*100:.2f}%")
    else:
        print("Algorithm 1 (Random Forest): skipped")
    if acc2_test is not None:
        print(f"Algorithm 2 (Gradient Boosting):  Train: {acc2_train*100:.2f}% | Test: {acc2_test*100:.2f}%")
    else:
        print("Algorithm 2 (Gradient Boosting): skipped")
    
    # Determine best model based on available test accuracy
    best_model = None
    best_name = None
    best_type = None
    if acc1_test is not None and (acc2_test is None or acc1_test >= acc2_test):
        print(f"\n[BEST] Algorithm 1 (Random Forest) - {acc1_test*100:.2f}% test accuracy")
        best_model = algo1_model
        best_name = "algorithm_1_random_forest"
        best_type = "random_forest"
    elif acc2_test is not None:
        print(f"\n[BEST] Algorithm 2 (Gradient Boosting) - {acc2_test*100:.2f}% test accuracy")
        best_model = algo2_model
        best_name = "algorithm_2_gradient_boosting"
        best_type = "gradient_boosting"
    else:
        print("\nNo models were trained.")

    print("="*60)
    
    return {
        "algo1_model": algo1_model,
        "algo1_acc_train": acc1_train,
        "algo1_acc_test": acc1_test,
        "algo2_model": algo2_model,
        "algo2_acc_train": acc2_train,
        "algo2_acc_test": acc2_test,
        "best_model": best_model,
        "best_name": best_name,
        "best_type": best_type,
        "scaler": scaler
    }

def save_models(results):
    """Save trained models and comparison data"""
    print("\nSaving models...")
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save Algorithm 1 if trained
    if results["algo1_model"] is not None:
        with open("models/algorithm_1_rf.pkl", "wb") as f:
            pickle.dump(results["algo1_model"], f)
        print("[OK] Algorithm 1 (Random Forest) saved")
    else:
        print("[SKIP] Algorithm 1 (Random Forest) not trained")
    
    # Save Algorithm 2 if trained
    if results["algo2_model"] is not None:
        with open("models/algorithm_2_gb.pkl", "wb") as f:
            pickle.dump(results["algo2_model"], f)
        print("[OK] Algorithm 2 (Gradient Boosting) saved")
    else:
        print("[SKIP] Algorithm 2 (Gradient Boosting) not trained")
    
    # Save scaler
    with open("models/feature_scaler.pkl", "wb") as f:
        pickle.dump(results["scaler"], f)
    print("[OK] Feature scaler saved")
    
    # Save best model
    if results['best_model'] is not None and results['best_name']:
        with open(f"models/{results['best_name']}_best.pkl", "wb") as f:
            pickle.dump(results["best_model"], f)
        print(f"[OK] Best model ({results['best_name']}) saved")
    else:
        print("[SKIP] No best model to save")
    
    # Save comparison data with only trained models
    comparison = {}
    if results["algo1_model"] is not None:
        comparison["algorithm_1_random_forest"] = {
            "name": "Random Forest",
            "train_accuracy": float(results["algo1_acc_train"]),
            "test_accuracy": float(results["algo1_acc_test"]),
            "type": "random_forest"
        }
    if results["algo2_model"] is not None:
        comparison["algorithm_2_gradient_boosting"] = {
            "name": "Gradient Boosting",
            "train_accuracy": float(results["algo2_acc_train"]),
            "test_accuracy": float(results["algo2_acc_test"]),
            "type": "gradient_boosting"
        }
    comparison["best_model"] = results["best_name"]
    if results["best_name"]:
        comparison["best_model_name"] = "Random Forest" if results["best_name"] == "algorithm_1_random_forest" else "Gradient Boosting"
    else:
        comparison["best_model_name"] = ""
    
    with open("models/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print("[OK] Comparison results saved to models/comparison.json")
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    if results['algo1_acc_test'] is not None:
        print(f"  Algorithm 1 (Random Forest):      {results['algo1_acc_test']*100:.2f}%")
    if results['algo2_acc_test'] is not None:
        print(f"  Algorithm 2 (Gradient Boosting):  {results['algo2_acc_test']*100:.2f}%")
    if comparison.get('best_model_name'):
        print(f"  Best Model: {comparison['best_model_name']}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Plant Disease Recognition Models')
    parser.add_argument('--dataset', type=str, default=DATASET_PATH, 
                        help='Path to the plant disease dataset folder')
    parser.add_argument('--max-samples', type=int, default=MAX_SAMPLES_PER_CLASS,
                        help='Maximum samples per class (default: 500)')
    parser.add_argument('--only-gb', action='store_true', help='Train only Gradient Boosting (skip Random Forest)')
    args = parser.parse_args()
    
    DATASET_PATH = args.dataset
    MAX_SAMPLES_PER_CLASS = args.max_samples
    
    logger.info("="*60)
    logger.info("PLANT DISEASE RECOGNITION - ML ALGORITHM COMPARISON")
    logger.info("Training on Real Dataset")
    logger.info("="*60)
    logger.info(f"Dataset Path: {DATASET_PATH}")
    logger.info(f"Max Samples Per Class: {MAX_SAMPLES_PER_CLASS}")
    
    # Load dataset
    X, y, class_names, class_to_idx = load_dataset_from_folder(DATASET_PATH)
    
    if X is None:
        print("\nERROR: Could not load dataset. Please check the DATASET_PATH.")
        exit(1)
    
    # Split dataset
    print("\nSplitting dataset into train/test sets...")
    print(f"Train size: {TRAIN_SIZE*100}% | Test size: {TEST_SIZE*100}%")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train models (optionally skip Algorithm 1)
    results = train_models(
        X_train, X_test, y_train, y_test,
        train_algo1=not args.only_gb,
        train_algo2=True
    )
    
    # Save models
    save_models(results)
    
    print("\n[SUCCESS] All models trained and saved!")
    print("You can now run app.py to use the trained models.")

