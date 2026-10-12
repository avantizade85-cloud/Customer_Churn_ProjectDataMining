from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file
)

import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path
from database import create_table, get_connection

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =========================================================
# 1. FLASK APPLICATION
# =========================================================

app = Flask(__name__)

create_table()


# =========================================================
# 2. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_DIR = PROJECT_DIR / "Models"
DATASET_DIR = PROJECT_DIR / "Dataset"
OUTPUT_DIR = PROJECT_DIR / "Outputs"

MODEL_PATH = MODEL_DIR / "best_churn_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "churn_preprocessor.pkl"

KMEANS_MODEL_PATH = (
    MODEL_DIR / "customer_segmentation_kmeans.pkl"
)

KMEANS_SCALER_PATH = (
    MODEL_DIR / "kmeans_scaler.pkl"
)

MODEL_INFO_PATH = MODEL_DIR / "model_info.json"

DATA_PATH = DATASET_DIR / "telco_customer_churn.csv"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 3. CHECK REQUIRED FILES
# =========================================================

required_files = [
    MODEL_PATH,
    PREPROCESSOR_PATH,
    KMEANS_MODEL_PATH,
    KMEANS_SCALER_PATH,
    MODEL_INFO_PATH,
    DATA_PATH
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required project file not found: {file_path}"
        )


# =========================================================
# 4. LOAD SAVED MODELS
# =========================================================

model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

kmeans_model = joblib.load(
    KMEANS_MODEL_PATH
)

kmeans_scaler = joblib.load(
    KMEANS_SCALER_PATH
)


# =========================================================
# 5. LOAD MODEL INFORMATION
# =========================================================

with open(MODEL_INFO_PATH, "r") as file:

    model_info = json.load(file)


# =========================================================
# 6. DATASET FEATURE COLUMNS
# =========================================================

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


# =========================================================
# 7. HELPER FUNCTIONS
# =========================================================

def load_dataset():

    """
    Load the original customer dataset.
    """

    dataset = pd.read_csv(
        DATA_PATH,
        dtype={"customerID": str}
    )

    dataset["TotalCharges"] = pd.to_numeric(
        dataset["TotalCharges"],
        errors="coerce"
    )

    return dataset


def clean_dataset(dataset):

    """
    Remove records with missing required values
    and duplicate rows.
    """

    required_columns = FEATURE_COLUMNS + ["Churn"]

    cleaned = dataset.dropna(
        subset=required_columns
    ).copy()

    cleaned = cleaned.drop_duplicates().copy()

    cleaned["Churn"] = cleaned["Churn"].astype(str).str.strip()

    cleaned = cleaned[
        cleaned["Churn"].isin(["Yes", "No"])
    ].copy()

    return cleaned


def get_churn_probabilities(processed_data):

    """
    Return the probability of class 1,
    where Churn = Yes.
    """

    if not hasattr(model, "predict_proba"):

        raise ValueError(
            "The saved model does not support predict_proba."
        )

    probabilities = model.predict_proba(
        processed_data
    )

    classes = list(model.classes_)

    if 1 not in classes:

        raise ValueError(
            "The saved model does not contain the churn class 1."
        )

    churn_index = classes.index(1)

    return probabilities[:, churn_index]


def get_risk_level(probability):

    """
    Convert churn probability into risk categories.
    """

    probability_percent = probability * 100

    if probability_percent >= 70:

        return "High"

    elif probability_percent >= 40:

        return "Medium"

    return "Low"


def get_recommendation(risk_level):

    """
    Generate a basic retention recommendation.
    """

    if risk_level == "High":

        return (
            "Prioritize this customer for retention. "
            "Review contract options, service concerns, "
            "and personalized retention offers."
        )

    elif risk_level == "Medium":

        return (
            "Monitor customer engagement and consider "
            "appropriate service or contract offers."
        )

    return (
        "Maintain service quality and continue "
        "regular customer engagement."
    )


def save_prediction_to_database(data, result):

    """
    Store an individual prediction in SQLite.
    """

    connection = get_connection()

    try:

        connection.execute(
            """
            INSERT INTO predictions (
                gender,
                senior_citizen,
                partner,
                dependents,
                tenure,
                phone_service,
                multiple_lines,
                internet_service,
                contract,
                payment_method,
                monthly_charges,
                total_charges,
                prediction,
                probability,
                risk_level,
                recommendation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(data["gender"]),
                int(data["SeniorCitizen"]),
                str(data["Partner"]),
                str(data["Dependents"]),
                float(data["tenure"]),
                str(data["PhoneService"]),
                str(data["MultipleLines"]),
                str(data["InternetService"]),
                str(data["Contract"]),
                str(data["PaymentMethod"]),
                float(data["MonthlyCharges"]),
                float(data["TotalCharges"]),
                result["prediction"],
                float(result["probability"]),
                result["risk_level"],
                result["recommendation"]
            )
        )

        connection.commit()

    finally:

        connection.close()


# =========================================================
# 8. LOAD DATA FOR THE DASHBOARD
# =========================================================

df = load_dataset()


# =========================================================
# 9. HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# 10. DASHBOARD API
# =========================================================

@app.route("/api/dashboard")
def dashboard():

    try:

        total_customers = int(len(df))

        churned_customers = int(
            (df["Churn"] == "Yes").sum()
        )

        non_churned_customers = int(
            (df["Churn"] == "No").sum()
        )

        churn_rate = 0.0

        if total_customers > 0:

            churn_rate = round(
                churned_customers / total_customers * 100,
                2
            )

        average_monthly_charges = round(
            float(df["MonthlyCharges"].mean()),
            2
        )

        return jsonify({

            "success": True,

            "total_customers": total_customers,

            "churned_customers": churned_customers,

            "non_churned_customers": non_churned_customers,

            "churn_rate": churn_rate,

            "average_monthly_charges": average_monthly_charges

        })

    except Exception as error:

        app.logger.exception("Dashboard API failed")

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# =========================================================
# 11. MODEL INFORMATION API
# =========================================================

@app.route("/api/model-info")
def get_model_info():

    return jsonify(model_info)


# =========================================================
# 12. INDIVIDUAL CUSTOMER PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True)

        if not data:

            return jsonify({

                "success": False,

                "error": "No customer data received."

            }), 400

        # -----------------------------------------
        # Validate required input fields
        # -----------------------------------------

        missing_fields = [
            column
            for column in FEATURE_COLUMNS
            if column not in data
        ]

        if missing_fields:

            return jsonify({

                "success": False,

                "error": (
                    "Missing required fields: "
                    + ", ".join(missing_fields)
                )

            }), 400

        # -----------------------------------------
        # Build customer DataFrame
        # -----------------------------------------

        customer_data = pd.DataFrame([{

            "gender": str(data["gender"]),

            "SeniorCitizen": int(data["SeniorCitizen"]),

            "Partner": str(data["Partner"]),

            "Dependents": str(data["Dependents"]),

            "tenure": float(data["tenure"]),

            "PhoneService": str(data["PhoneService"]),

            "MultipleLines": str(data["MultipleLines"]),

            "InternetService": str(data["InternetService"]),

            "OnlineSecurity": str(data["OnlineSecurity"]),

            "OnlineBackup": str(data["OnlineBackup"]),

            "DeviceProtection": str(data["DeviceProtection"]),

            "TechSupport": str(data["TechSupport"]),

            "StreamingTV": str(data["StreamingTV"]),

            "StreamingMovies": str(data["StreamingMovies"]),

            "Contract": str(data["Contract"]),

            "PaperlessBilling": str(data["PaperlessBilling"]),

            "PaymentMethod": str(data["PaymentMethod"]),

            "MonthlyCharges": float(data["MonthlyCharges"]),

            "TotalCharges": float(data["TotalCharges"])

        }])

        # -----------------------------------------
        # Apply saved preprocessing
        # -----------------------------------------

        processed_data = preprocessor.transform(
            customer_data
        )

        # -----------------------------------------
        # Predict churn
        # -----------------------------------------

        prediction = int(
            model.predict(processed_data)[0]
        )

        churn_probability = float(
            get_churn_probabilities(processed_data)[0]
        )

        probability_percent = round(
            churn_probability * 100,
            2
        )

        churn_result = (
            "Yes" if prediction == 1 else "No"
        )

        risk_level = get_risk_level(
            churn_probability
        )

        recommendation = get_recommendation(
            risk_level
        )

        result = {

            "success": True,

            "prediction": churn_result,

            "probability": probability_percent,

            "raw_probability": churn_probability,

            "risk_level": risk_level,

            "recommendation": recommendation

        }

        # -----------------------------------------
        # Store prediction
        # -----------------------------------------

        save_prediction_to_database(
            data,
            result
        )

        print("\nIndividual prediction completed.")

        print("Prediction:", churn_result)

        print("Raw churn probability:", churn_probability)

        print("Churn probability (%):", probability_percent)

        print("Risk:", risk_level)

        print("Prediction saved to SQLite.")

        return jsonify(result)

    except Exception as error:

        app.logger.exception("Individual prediction failed")

        return jsonify({

            "success": False,

            "error": str(error)

        }), 400


# =========================================================
# 13. DATASET-BASED PREDICTION
# =========================================================

@app.route("/predict-dataset", methods=["POST"])
def predict_dataset():

    try:

        # -----------------------------------------
        # Load and clean provided CSV
        # -----------------------------------------

        dataset = clean_dataset(
            load_dataset()
        )

        if dataset.empty:

            return jsonify({

                "success": False,

                "error": "The dataset has no usable customer rows."

            }), 400

        X = dataset[FEATURE_COLUMNS].copy()

        # Actual churn labels are used for evaluation only.
        # They are NOT included as model input features.

        y = dataset["Churn"].map({

            "No": 0,

            "Yes": 1

        })

        if y.nunique() < 2:

            return jsonify({

                "success": False,

                "error": (
                    "The dataset needs both churn classes "
                    "(Yes and No) for evaluation."
                )

            }), 400

        # -----------------------------------------
        # Predict for every usable customer row
        # -----------------------------------------

        processed_all = preprocessor.transform(X)

        predictions = model.predict(
            processed_all
        )

        churn_probabilities = get_churn_probabilities(
            processed_all
        )

        probabilities_percent = np.round(
            churn_probabilities * 100,
            2
        )

        risk_levels = [
            get_risk_level(float(probability))
            for probability in churn_probabilities
        ]

        recommendations = [
            get_recommendation(risk)
            for risk in risk_levels
        ]

        # -----------------------------------------
        # Create output for the entire dataset
        # -----------------------------------------

        output_df = pd.DataFrame({

            "customerID": dataset["customerID"].astype(str).values,

            "tenure": dataset["tenure"].values,

            "Contract": dataset["Contract"].values,

            "InternetService": dataset["InternetService"].values,

            "MonthlyCharges": dataset["MonthlyCharges"].values,

            "TotalCharges": dataset["TotalCharges"].values,

            "Actual_Churn": dataset["Churn"].values,

            "Predicted_Churn": np.where(
                predictions == 1,
                "Yes",
                "No"
            ),

            "Churn_Probability_Percent": probabilities_percent,

            "Risk_Level": risk_levels,

            "Recommendation": recommendations

        })

        # -----------------------------------------
        # Save full prediction output
        # -----------------------------------------

        output_path = (
            OUTPUT_DIR / "dataset_predictions.csv"
        )

        output_df.to_csv(
            output_path,
            index=False
        )

        # -----------------------------------------
        # Evaluate using the held-out test split
        # -----------------------------------------
        #
        # This reproduces the 80/20 split, provided
        # the saved model was trained on this same
        # cleaned dataset and split configuration.
        #
        # Do not use metrics calculated on all rows
        # as an unbiased test score.

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=42,

            stratify=y

        )

        processed_test = preprocessor.transform(
            X_test
        )

        test_predictions = model.predict(
            processed_test
        )

        test_probabilities = get_churn_probabilities(
            processed_test
        )

        test_accuracy = accuracy_score(
            y_test,
            test_predictions
        )

        test_precision = precision_score(
            y_test,
            test_predictions,
            zero_division=0
        )

        test_recall = recall_score(
            y_test,
            test_predictions,
            zero_division=0
        )

        test_f1 = f1_score(
            y_test,
            test_predictions,
            zero_division=0
        )

        test_roc_auc = roc_auc_score(
            y_test,
            test_probabilities
        )

        # -----------------------------------------
        # Summary
        # -----------------------------------------

        predicted_churn = int(
            (predictions == 1).sum()
        )

        predicted_no_churn = int(
            (predictions == 0).sum()
        )

        high_risk_count = int(
            sum(risk == "High" for risk in risk_levels)
        )

        medium_risk_count = int(
            sum(risk == "Medium" for risk in risk_levels)
        )

        low_risk_count = int(
            sum(risk == "Low" for risk in risk_levels)
        )

        result = {

            "success": True,

            "rows_processed": int(len(output_df)),

            "predicted_churn": predicted_churn,

            "predicted_no_churn": predicted_no_churn,

            "high_risk_customers": high_risk_count,

            "medium_risk_customers": medium_risk_count,

            "low_risk_customers": low_risk_count,

            "test_metrics": {

                "accuracy": round(
                    float(test_accuracy) * 100,
                    2
                ),

                "precision": round(
                    float(test_precision) * 100,
                    2
                ),

                "recall": round(
                    float(test_recall) * 100,
                    2
                ),

                "f1_score": round(
                    float(test_f1) * 100,
                    2
                ),

                "roc_auc": round(
                    float(test_roc_auc) * 100,
                    2
                )

            },

            "download_url": "/download-predictions"

        }

        print("\n========================================")

        print("DATASET PREDICTION COMPLETED")

        print("Rows processed:", len(output_df))

        print("Predicted churn:", predicted_churn)

        print("Predicted no churn:", predicted_no_churn)

        print("High-risk customers:", high_risk_count)

        print("Saved output:", output_path)

        print("Held-out test metrics:", result["test_metrics"])

        print("========================================\n")

        return jsonify(result)

    except Exception as error:

        app.logger.exception("Dataset prediction failed")

        return jsonify({

            "success": False,

            "error": str(error)

        }), 400


# =========================================================
# 14. DOWNLOAD DATASET PREDICTIONS
# =========================================================

@app.route("/download-predictions")
def download_predictions():

    output_path = (
        OUTPUT_DIR / "dataset_predictions.csv"
    )

    if not output_path.exists():

        return jsonify({

            "success": False,

            "error": (
                "Run dataset prediction before downloading "
                "the prediction results."
            )

        }), 404

    return send_file(

        output_path,

        as_attachment=True,

        download_name="dataset_predictions.csv",

        mimetype="text/csv"

    )


# =========================================================
# 15. CUSTOMER SEGMENTATION API
# =========================================================

@app.route("/segment", methods=["POST"])
def segment_customer():

    try:

        data = request.get_json(silent=True)

        if not data:

            return jsonify({

                "success": False,

                "error": "No customer information received."

            }), 400

        tenure = float(
            data["tenure"]
        )

        monthly_charges = float(
            data["MonthlyCharges"]
        )

        total_charges = float(
            data["TotalCharges"]
        )

        segment_data = np.array([[

            tenure,

            monthly_charges,

            total_charges

        ]])

        scaled_segment = kmeans_scaler.transform(
            segment_data
        )

        segment = int(
            kmeans_model.predict(
                scaled_segment
            )[0]
        )

        return jsonify({

            "success": True,

            "segment": segment

        })

    except Exception as error:

        app.logger.exception("Segmentation failed")

        return jsonify({

            "success": False,

            "error": str(error)

        }), 400


# =========================================================
# 16. PREDICTION HISTORY API
# =========================================================

@app.route("/api/prediction-history")
def prediction_history():

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT
                id,
                gender,
                tenure,
                contract,
                monthly_charges,
                prediction,
                probability,
                risk_level,
                recommendation,
                created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT 100
            """
        ).fetchall()

        columns = [

            "id",
            "gender",
            "tenure",
            "contract",
            "monthly_charges",
            "prediction",
            "probability",
            "risk_level",
            "recommendation",
            "created_at"

        ]

        records = [

            dict(zip(columns, row))
            for row in rows

        ]

        return jsonify({

            "success": True,

            "count": len(records),

            "history": records

        })

    except Exception as error:

        app.logger.exception("Prediction history request failed")

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500

    finally:

        connection.close()


# =========================================================
# 17. RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    print("========================================")

    print("Customer Churn Web Application")

    print("Starting Flask server...")

    print("Dataset:", DATA_PATH)

    print("========================================")

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )