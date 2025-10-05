import gradio as gr
import numpy as np
import json
import joblib
import os
import sys
import subprocess
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import pandas as pd
# --------------------------
# Scheduled Task Functions
# --------------------------
def run_kepler_training():
    """
    Runs Kepler_model.py every 24 hours
    """
    try:
        print(f"[CRON-KEPLER] Starting scheduled model training at {datetime.now()}")
        
        result = subprocess.run(
            [sys.executable, "Kepler_model.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            print(f"[CRON-KEPLER] Training completed successfully at {datetime.now()}")
            print(f"[CRON-KEPLER] Output: {result.stdout}")
        else:
            print(f"[CRON-KEPLER] Training failed with error code {result.returncode}")
            print(f"[CRON-KEPLER] Error: {result.stderr}")
            
    except Exception as e:
        print(f"[CRON-KEPLER] Exception occurred: {e}")


def run_k2_training():
    """
    Runs K2_model.py every 24 hours
    """
    try:
        print(f"[CRON-K2] Starting scheduled model training at {datetime.now()}")
        
        result = subprocess.run(
            [sys.executable, "K2_model.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode == 0:
            print(f"[CRON-K2] Training completed successfully at {datetime.now()}")
            print(f"[CRON-K2] Output: {result.stdout}")
        else:
            print(f"[CRON-K2] Training failed with error code {result.returncode}")
            print(f"[CRON-K2] Error: {result.stderr}")
            
    except Exception as e:
        print(f"[CRON-K2] Exception occurred: {e}")


# --------------------------
# Initialize Scheduler
# --------------------------
scheduler = BackgroundScheduler()

# Schedule Kepler model training
scheduler.add_job(
    func=run_kepler_training,
    trigger="interval",
    hours=24,
    id="kepler_training_job",
    name="Run Kepler_model.py every 24 hours",
    replace_existing=True
)

# Schedule K2 model training
scheduler.add_job(
    func=run_k2_training,
    trigger="interval",
    hours=24,
    id="k2_training_job",
    name="Run K2_model.py every 24 hours",
    replace_existing=True
)

scheduler.start()
print(f"[INFO] Scheduler started.")
for job in scheduler.get_jobs():
    print(f"[INFO] {job.name} - Next run at: {job.next_run_time}")

# --------------------------
# Load Models and Encoders
# --------------------------
def load_artifact(path):
    """Helper to load artifacts and print their type"""
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return None

    try:
        obj = joblib.load(path)
        print(f"[INFO] Loaded {path}, type: {type(obj)}")
        return obj
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
        return None


# Load Kepler model
print("\n[INFO] Loading Kepler model...")
kepler_model = load_artifact("decision_tree_kepler.pkl")
kepler_encoder = load_artifact("label_encoder_kepler.pkl")

# Load K2 model
print("\n[INFO] Loading K2 model...")
k2_model = load_artifact("decision_tree_k2.pkl")
k2_encoder = load_artifact("label_encoder_k2.pkl")

# Validate models
def validate_model(model, name):
    """Validate that model is properly loaded"""
    if model is None:
        print(f"[WARNING] {name} model not loaded")
        return False
    
    if not hasattr(model, "predict"):
        print(f"[ERROR] {name} model is not a valid sklearn estimator")
        return False
    
    try:
        n_features = model.n_features_in_
        print(f"[INFO] {name} model expects {n_features} features")
        return True
    except AttributeError:
        print(f"[ERROR] {name} model does not have `n_features_in_`")
        return False

kepler_valid = validate_model(kepler_model, "Kepler")
k2_valid = validate_model(k2_model, "K2")

# Get feature counts
kepler_features = kepler_model.n_features_in_ if kepler_valid else 0
k2_features = k2_model.n_features_in_ if k2_valid else 0

# --------------------------
# Prediction Functions
# --------------------------
def predict_kepler(input_vector):
    """
    Predicts using Kepler model
    """
    if not kepler_valid:
        return json.dumps({
            "success": False,
            "error": "Kepler model not available",
            "message": "The Kepler model is not properly loaded"
        }, indent=2)
    
    try:
        # Handle string input
        if isinstance(input_vector, str):
            input_vector = [float(x.strip()) for x in input_vector.split(',')]

        # Convert to numpy array
        input_array = np.array(input_vector).reshape(1, -1)

        # Validate feature count
        if input_array.shape[1] != kepler_features:
            raise ValueError(f"Expected {kepler_features} features, but got {input_array.shape[1]}")

        # Make prediction
        prediction = kepler_model.predict(input_array)
        prediction_proba = kepler_model.predict_proba(input_array)
        predicted_label = kepler_encoder.inverse_transform(prediction)[0]

        # Probabilities for each class
        class_probabilities = {}
        for idx, prob in enumerate(prediction_proba[0]):
            class_name = kepler_encoder.inverse_transform([idx])[0]
            class_probabilities[str(class_name)] = float(prob)

        response = {
            "success": True,
            "model": "Kepler",
            "prediction": str(predicted_label),
            "confidence": float(max(prediction_proba[0])),
            "all_probabilities": class_probabilities,
            "input_features": len(input_vector)
        }
        return json.dumps(response, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "model": "Kepler",
            "error": str(e),
            "message": f"Please provide {kepler_features} comma-separated values"
        }, indent=2)


def predict_k2(input_vector):
    """
    Predicts using K2 model
    """
    if not k2_valid:
        return json.dumps({
            "success": False,
            "error": "K2 model not available",
            "message": "The K2 model is not properly loaded"
        }, indent=2)
    
    try:
        # Handle string input
        if isinstance(input_vector, str):
            input_vector = [float(x.strip()) for x in input_vector.split(',')]

        # Convert to numpy array
        input_array = np.array(input_vector).reshape(1, -1)

        # Validate feature count
        if input_array.shape[1] != k2_features:
            raise ValueError(f"Expected {k2_features} features, but got {input_array.shape[1]}")

        # Make prediction
        prediction = k2_model.predict(input_array)
        prediction_proba = k2_model.predict_proba(input_array)
        predicted_label = k2_encoder.inverse_transform(prediction)[0]

        # Probabilities for each class
        class_probabilities = {}
        for idx, prob in enumerate(prediction_proba[0]):
            class_name = k2_encoder.inverse_transform([idx])[0]
            class_probabilities[str(class_name)] = float(prob)

        response = {
            "success": True,
            "model": "K2",
            "prediction": str(predicted_label),
            "confidence": float(max(prediction_proba[0])),
            "all_probabilities": class_probabilities,
            "input_features": len(input_vector)
        }
        return json.dumps(response, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "model": "K2",
            "error": str(e),
            "message": f"Please provide {k2_features} comma-separated values"
        }, indent=2)


# --------------------------
# Kepler Info Functions
# --------------------------
def get_kepler_info(kepler_id):
    """
    Given a Kepler ID, retrieves ALL planetary and stellar data
    from Kepler_info.csv and returns it as JSON.
    """
    try:
        # Load the CSV (adjust path if needed)
        file_path = "Kepler_info.csv"
        if not os.path.exists(file_path):
            return json.dumps({
                "success": False,
                "error": f"File not found: {file_path}"
            }, indent=2)

        df = pd.read_csv(file_path)

        # Try to match kepler_id column
        id_col = None
        for possible in ["Kepler_ID"]:
            if possible in df.columns:
                id_col = possible
                break

        if id_col is None:
            return json.dumps({
                "success": False,
                "error": "No Kepler ID column found (expected 'kepid' or 'kepler_id')"
            }, indent=2)

        # Filter row by Kepler ID
        row = df[df[id_col] == int(kepler_id)]
        if row.empty:
            return json.dumps({
                "success": False,
                "error": f"No planet found with Kepler ID {kepler_id}"
            }, indent=2)

        # Convert the entire row to a dictionary
        record = row.iloc[0].to_dict()
        
        # Convert numpy/pandas types to native Python types for JSON serialization
        cleaned_record = {}
        for k, v in record.items():
            if pd.isna(v):
                cleaned_record[k] = None
            elif isinstance(v, (int, float)):
                cleaned_record[k] = float(v) if isinstance(v, float) else int(v)
            else:
                cleaned_record[k] = str(v)

        return json.dumps({
            "success": True,
            "kepler_id": int(kepler_id),
            "data": cleaned_record
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


def get_kepler_paginated(page, page_size):
    """
    Retrieves paginated data from Kepler_info.csv
    
    Args:
        page: Page number (starting from 1)
        page_size: Number of records per page
    
    Returns:
        JSON with paginated results and metadata
    """
    try:
        # Convert inputs to integers
        page = int(page)
        page_size = int(page_size)
        
        # Validate inputs
        if page < 1:
            return json.dumps({
                "success": False,
                "error": "Page number must be >= 1"
            }, indent=2)
        
        if page_size < 1 or page_size > 1000:
            return json.dumps({
                "success": False,
                "error": "Page size must be between 1 and 1000"
            }, indent=2)
        
        # Load the CSV
        file_path = "Kepler_info.csv"
        if not os.path.exists(file_path):
            return json.dumps({
                "success": False,
                "error": f"File not found: {file_path}"
            }, indent=2)

        df = pd.read_csv(file_path)
        total_records = len(df)
        total_pages = (total_records + page_size - 1) // page_size  # Ceiling division
        
        # Check if page exists
        if page > total_pages and total_records > 0:
            return json.dumps({
                "success": False,
                "error": f"Page {page} does not exist. Total pages: {total_pages}"
            }, indent=2)
        
        # Calculate start and end indices
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_records)
        
        # Get the page of data
        page_data = df.iloc[start_idx:end_idx]
        
        # Convert to list of dictionaries with cleaned data
        records = []
        for _, row in page_data.iterrows():
            cleaned_record = {}
            for k, v in row.items():
                if pd.isna(v):
                    cleaned_record[k] = None
                elif isinstance(v, (int, float)):
                    cleaned_record[k] = float(v) if isinstance(v, float) else int(v)
                else:
                    cleaned_record[k] = str(v)
            records.append(cleaned_record)
        
        # Build response
        response = {
            "success": True,
            "pagination": {
                "current_page": page,
                "page_size": page_size,
                "total_records": total_records,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1,
                "records_on_page": len(records)
            },
            "data": records
        }
        
        return json.dumps(response, indent=2)
    
    except ValueError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid input: {str(e)}"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


# --------------------------
# Gradio Interfaces
# --------------------------

# Kepler Interface
kepler_interface = gr.Interface(
    fn=predict_kepler,
    inputs=gr.Textbox(
        label="Input Vector",
        placeholder=f"Enter {kepler_features} comma-separated values",
        lines=2
    ),
    outputs=gr.JSON(label="Kepler Prediction Result"),
    title="🔭 Kepler Model Predictions",
    description=f"Enter your feature vector with **{kepler_features} values** to predict exoplanet disposition using the Kepler dataset model.",
    examples=[
        [", ".join(["1.0"] * kepler_features)] if kepler_valid else [],
    ],
    api_name="predict_kepler"
)

# K2 Interface
k2_interface = gr.Interface(
    fn=predict_k2,
    inputs=gr.Textbox(
        label="Input Vector",
        placeholder=f"Enter {k2_features} comma-separated values",
        lines=2
    ),
    outputs=gr.JSON(label="K2 Prediction Result"),
    title="🛰️ K2 Model Predictions",
    description=f"Enter your feature vector with **{k2_features} values** to predict exoplanet disposition using the K2 dataset model.",
    examples=[
        [", ".join(["1.0"] * k2_features)] if k2_valid else [],
    ],
    api_name="predict_k2"
)

# Kepler Info Lookup Interface
kepler_info_interface = gr.Interface(   
    fn=get_kepler_info,
    inputs=gr.Textbox(label="Enter Kepler ID", placeholder="e.g. 2304168"),
    outputs=gr.JSON(label="Kepler Planet Info"),
    title="🌠 Kepler Planet Data Lookup",
    description="Retrieve ALL stellar and planetary data from Kepler_info.csv by Kepler ID.",
    api_name="keplerid"
)

# Paginated Kepler Data Interface
kepler_paginated_interface = gr.Interface(
    fn=get_kepler_paginated,
    inputs=[
        gr.Number(label="Page Number", value=1, minimum=1, precision=0),
        gr.Number(label="Page Size", value=10, minimum=1, maximum=1000, precision=0)
    ],
    outputs=gr.JSON(label="Paginated Kepler Data"),
    title="📄 Kepler Data Pagination",
    description="Retrieve multiple records from Kepler_info.csv with pagination. Page size limited to 1000 records per page.",
    examples=[
        [1, 10],
        [1, 50],
        [2, 20]
    ],
    api_name="kepler_paginated"
)

# --------------------------
# Combined App with Tabs
# --------------------------
app = gr.TabbedInterface(
    [kepler_interface, k2_interface, kepler_info_interface, kepler_paginated_interface],
    ["Kepler Model", "K2 Model", "Kepler Info Lookup", "Kepler Paginated Data"],
    title="🌌 NASA Exoplanet Detection API",
    theme=gr.themes.Soft()
)


# --------------------------
# Launch App
# --------------------------
if __name__ == "__main__":
    try:
        app.launch()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("[INFO] Scheduler shut down gracefully")