import warnings
import numpy as np
import pandas as pd
import joblib
import hashlib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# -------------------------------
# Global Variables
# -------------------------------
FILE_PATH = "K2_data.csv"
MODEL_PATH = "decision_tree_k2.pkl"
ENCODER_PATH = "label_encoder_k2.pkl"
FEATURES_PATH = "feature_names_k2.pkl"
HASH_FILE = "k2_data_hash.txt"

# -------------------------------
# Hash Management Functions
# -------------------------------
def calculate_file_hash(file_path):
    """
    Calculate SHA256 hash of a file
    """
    sha256_hash = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"[ERROR] Failed to calculate hash: {e}")
        return None


def load_stored_hash(hash_file):
    """
    Load the stored hash from file
    Returns None if file doesn't exist
    """
    if not os.path.exists(hash_file):
        print(f"[INFO] Hash file not found: {hash_file}")
        return None
    
    try:
        with open(hash_file, "r") as f:
            stored_hash = f.read().strip()
        print(f"[INFO] Loaded stored hash: {stored_hash[:16]}...")
        return stored_hash
    except Exception as e:
        print(f"[ERROR] Failed to read hash file: {e}")
        return None


def save_hash(hash_value, hash_file):
    """
    Save hash to file
    """
    try:
        with open(hash_file, "w") as f:
            f.write(hash_value)
        print(f"[INFO] Saved new hash to {hash_file}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save hash: {e}")
        return False


def check_if_data_changed(file_path, hash_file):
    """
    Check if data file has changed since last training
    Returns: (has_changed, current_hash)
    """
    current_hash = calculate_file_hash(file_path)
    
    if current_hash is None:
        print("[ERROR] Could not calculate current hash")
        return True, None  # Assume changed to trigger training
    
    print(f"[INFO] Current data hash: {current_hash[:16]}...")
    
    stored_hash = load_stored_hash(hash_file)
    
    if stored_hash is None:
        print("[INFO] No previous hash found - this appears to be first training")
        return True, current_hash
    
    if current_hash == stored_hash:
        print("[INFO] Data hash matches - no changes detected")
        return False, current_hash
    else:
        print("[INFO] Data hash differs - changes detected")
        return True, current_hash


# -------------------------------
# Data Loading & Preprocessing
# -------------------------------
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Drop unused columns if present
    cols_to_drop = [
       "pl_bmassj", "pl_bmassprov", "pl_orbeccen", "pl_insol", "pl_eqt", "pl_trandep", 
       "pl_tranmid", "pl_trandur", "pl_occdep", "pl_orbtper", "pl_orblper", "pl_rvamp", 
       "pl_projobliq", "pl_trueobliq", "st_met", "st_metratio", "st_lum", "st_age", 
       "st_dens", "st_vsin", "st_rotp", "st_radv", "sy_pm", "sy_pmra", "sy_pmdec", 
       "sy_dist", "sy_plx", "sy_bmag", "sy_vmag", "sy_rmag", "sy_imag", "sy_zmag", 
       "sy_w1mag", "sy_w2mag", "sy_w3mag", "sy_w4mag", "sy_gaiamag", "sy_kepmag", 
       "rowupdate", "pl_pubdate", "releasedate", "pl_nnotes", "st_nnotes", "sy_nnotes", 
       "pl_nespec", "pl_ntran", "pl_nrv", "pl_npul", "pl_nptv", "pl_nast", "pl_nmic", 
       "pl_ndkin", "st_nspec", "st_nphot", "st_nrv", "st_nplc", "st_ntran", "st_nimg", 
       "sy_ndisk", "sy_nstar", "sy_nplanet", "sy_nrv", "sy_ntran", "sy_nspec", "sy_nphot"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Detect label column
    label_col = None
    for cand in ["disposition"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError("No label column found. Available columns: " + str(df.columns.tolist()))

    # Features and labels
    X = df.drop(columns=[label_col])
    y = df[label_col]
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Keep only numeric features
    X = X.select_dtypes(include=[np.number])
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=32)

    return X_train, X_val, y_train, y_val, label_encoder, X.columns.tolist()


# -------------------------------
# Decision Tree Training + Save
# -------------------------------
def train_and_save():
    X_train, X_val, y_train, y_val, label_encoder, feature_names = load_and_preprocess_data(FILE_PATH)

    # Train Decision Tree
    dt = DecisionTreeClassifier(
        criterion="gini",
        max_depth=10,  # Change to 5-6
        random_state=32
    )
    print("Columns kept:", len(X_train.columns))
    print("Column names:", X_train.columns.tolist())

    print("[INFO] Training Decision Tree model...")
    dt.fit(X_train, y_train)

    # Predictions on validation set
    dt_pred = dt.predict(X_val)

    # Evaluation
    print("\n🌳 Decision Tree Results")
    print("Accuracy:", accuracy_score(y_val, dt_pred))
    print(classification_report(y_val, dt_pred, target_names=label_encoder.classes_))

    # Save model + encoder + feature names
    joblib.dump(dt, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")
    print(f"[INFO] Label encoder saved to {ENCODER_PATH}")
    print(f"[INFO] Feature names saved to {FEATURES_PATH}")


# -------------------------------
# Main Execution with Hash Check
# -------------------------------
def main():
    print("=" * 60)
    print("🚀 Starting K2 Model Training Process")
    print("=" * 60)
    
    # Check if data file exists
    if not os.path.exists(FILE_PATH):
        print(f"[ERROR] Data file not found: {FILE_PATH}")
        return
    
    # Check if data has changed
    has_changed, current_hash = check_if_data_changed(FILE_PATH, HASH_FILE)
    
    if not has_changed:
        print("\n✅ Data unchanged - skipping training")
        print("=" * 60)
        return
    
    print("\n🔄 Data has changed - proceeding with training")
    print("=" * 60)
    
    try:
        # Train the model
        train_and_save()
        
        # Save the new hash only if training was successful
        if current_hash:
            save_hash(current_hash, HASH_FILE)
            print("\n✅ Training complete and hash updated")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("[INFO] Hash not updated - will retry on next run")
    
    print("=" * 60)


# -------------------------------
# Execute
# -------------------------------
if __name__ == "__main__":
    main()