"""
application.py
==============
Flask web application -- exposes three endpoints:

  GET  /        -> renders the prediction form (templates/index.html)
  POST /predict -> runs PredictPipeline and returns the result
  POST /train   -> re-trains the full pipeline and returns metrics

Environment variables (optional, set in .env or shell):
  HOST        Host to bind to (default: 127.0.0.1)
  PORT        Port to listen on (default: 5000)
  FLASK_DEBUG Enable debug mode -- set to "1" for development only

Run locally:
    python application.py
"""

import os
import sys

from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

from src.pipeline.prediction_pipeline import PredictPipeline, LeachingInput
from src.pipeline.training_pipeline   import TrainingPipeline
from src.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)
app = Flask(__name__)
CORS(app)  # allow cross-origin requests (e.g. from a separate frontend)


# -- Routes -------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Render the main prediction form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept form or JSON input, run inference, return result.

    Supports both HTML form submission and JSON API calls.
    """
    try:
        data = request.get_json() if request.is_json else request.form

        leaching_input = LeachingInput(
            Concentration_M=float(data["Concentration_M"]),
            Temperature_C=float(data["Temperature_C"]),
            Time_hrs=float(data["Time_hrs"]),
            SLR_gL=float(data["SLR_gL"]),
            Has_Reductant=int(data.get("Has_Reductant", 0)),
            Solvent_Type=str(data.get("Solvent_Type", "Organic Acid")),
            Battery_Chemistry_Std=str(data.get("Battery_Chemistry_Std", "LCO")),
            Reductant_Std=str(data.get("Reductant_Std", "None")),
            Target_Metal=str(data.get("Target_Metal", "Co")),
            RDKIT_MW=float(data.get("RDKIT_MW", 192.12)),
            RDKIT_LogP=float(data.get("RDKIT_LogP", -1.248)),
            RDKIT_TPSA=float(data.get("RDKIT_TPSA", 132.12)),
            RDKIT_HBD=int(data.get("RDKIT_HBD", 4)),
            RDKIT_HBA=int(data.get("RDKIT_HBA", 7)),
            RDKIT_RotBonds=int(data.get("RDKIT_RotBonds", 5)),
            RDKIT_HeavyAtoms=int(data.get("RDKIT_HeavyAtoms", 13)),
            RDKIT_Has_Carboxyl=int(data.get("RDKIT_Has_Carboxyl", 1)),
            RDKIT_Has_Hydroxyl=int(data.get("RDKIT_Has_Hydroxyl", 1)),
            RDKIT_Has_Halogen=int(data.get("RDKIT_Has_Halogen", 0)),
            RDKIT_Has_Phosphorus=int(data.get("RDKIT_Has_Phosphorus", 0)),
            RDKIT_Is_Ionic=int(data.get("RDKIT_Is_Ionic", 0)),
            RDKIT_Morgan_FP_Density=float(data.get("RDKIT_Morgan_FP_Density", 0.04)),
            EHS_Environment=float(data.get("EHS_Environment", 2.75)),
            EHS_Health=float(data.get("EHS_Health", 2.75)),
            EHS_Safety=float(data.get("EHS_Safety", 2.25)),
            EHS_Total=float(data.get("EHS_Total", 2.70)),
            GreenScore=float(data.get("GreenScore", 81.1)),
        )

        pipeline = PredictPipeline()
        prediction = pipeline.predict(leaching_input)

        if request.is_json:
            return jsonify({"predicted_efficiency_pct": prediction})

        return render_template(
            "index.html",
            prediction=f"{prediction:.2f}",
        )

    except FileNotFoundError as exc:
        # Artifacts missing -- give the user a clear action to take
        msg = str(exc)
        logger.error(msg)
        if request.is_json:
            return jsonify({"error": msg}), 503
        return render_template("index.html", error=msg)

    except (ValueError, KeyError) as exc:
        # Bad input -- 400, not 500
        msg = str(exc)
        logger.warning(f"Invalid input: {msg}")
        if request.is_json:
            return jsonify({"error": msg}), 400
        return render_template("index.html", error=msg)

    except Exception as exc:
        logger.error(f"Prediction error: {exc}", exc_info=True)
        if request.is_json:
            return jsonify({"error": str(exc)}), 500
        return render_template("index.html", error=str(exc))


@app.route("/train", methods=["POST"])
def train():
    """
    Re-train the full pipeline.
    Useful for CI/CD or re-training after new data arrives.
    """
    try:
        metrics = TrainingPipeline().run()
        return jsonify({"status": "success", "metrics": metrics})
    except Exception as exc:
        logger.error(f"Training error: {exc}", exc_info=True)
        return jsonify({"status": "error", "message": str(exc)}), 500


# -- Entry point --------------------------------------------------------------

if __name__ == "__main__":
    host  = os.environ.get("HOST", "127.0.0.1")
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    if debug:
        logger.warning("Running in DEBUG mode -- do not use in production.")

    logger.info(f"Starting Flask app on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
