# 🔬 Organic Acid Leaching Efficiency — ML Pipeline

End-to-end machine learning pipeline for predicting the **leaching efficiency (%)** of metals (Co, Li, Ni, Mn) from spent lithium-ion batteries using organic acid systems.

---

## 📁 Project Structure

```
ML_Leaching_Pipeline/
│
├── src/                          # Core Python package
│   ├── components/               # Modular pipeline stages
│   │   ├── data_ingestion.py     # Stage 1: Source-aware train/test split
│   │   ├── data_transformation.py# Stage 2: Feature engineering + preprocessing
│   │   ├── model_trainer.py      # Stage 3: Train & compare 5 models
│   │   └── model_evaluation.py   # Stage 4: Metrics + diagnostic plots
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py  # Orchestrates all 4 stages
│   │   └── prediction_pipeline.py# Inference: input → predicted efficiency %
│   │
│   └── utils/
│       ├── logger.py             # Centralised logging (file + console)
│       ├── exception.py          # Custom LeachingException with traceback
│       └── common.py             # save/load objects, evaluate models
│
├── artifacts/
│   ├── models/                   # preprocessor.joblib, best_model.joblib
│   ├── plots/                    # actual_vs_predicted.png, residuals.png
│   └── reports/                  # model_comparison.json, evaluation_report.json
│
├── logs/                         # Timestamped run logs
├── notebooks/                    # EDA notebooks (Jupyter)
├── templates/                    # Flask HTML templates
│   └── index.html
├── data/                         # Place raw_dataset.csv here
│
├── application.py                # Flask web app (GET / and POST /predict)
├── setup.py                      # Package installer
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone / download this folder
cd ML_Leaching_Pipeline

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in editable mode
pip install -e .
```

---

## 🚀 Usage

### 1 — Prepare Data

Copy your augmented labeled dataset CSV into `data/`:

```bash
cp Full_Augmented_Dataset_Labeled.csv data/raw_dataset.csv
```

### 2 — Run the Full Training Pipeline

```bash
python -m src.pipeline.training_pipeline
```

Or use the console script shortcut (after `pip install -e .`):

```bash
train
```

### 3 — Launch the Flask Web App

```bash
python application.py
```

Then open `http://localhost:5000` in your browser and enter leaching conditions to get a predicted efficiency.

### 4 — Run Individual Stages

```bash
# Stage 1 only
python -m src.components.data_ingestion

# Stage 3 only (requires artifacts from stages 1–2)
python -m src.components.model_trainer
```

---

## 🧠 ML Models Compared

| Model              | Notes                                           |
|--------------------|-------------------------------------------------|
| Random Forest      | Robust ensemble; handles nonlinearity           |
| XGBoost            | State-of-the-art tabular regression             |
| LightGBM           | Faster XGBoost; good for smaller datasets       |
| Ridge Regression   | Linear baseline with L2 regularisation          |
| SVR (RBF kernel)   | Strong generaliser on small real test sets      |

**Selection criterion**: highest Test R² on the **real-data test set only**  
(Synthetic rows are never used for evaluation.)

---

## 📊 Features Used

**Experimental:** Concentration, Temperature, Time, SLR, Has_Reductant  
**System:** Solvent Type, Battery Chemistry, Reductant Type, Target Metal  
**Molecular (RDKit):** MW, LogP, TPSA, HBD, HBA, RotBonds, fingerprint density, functional groups  
**Sustainability (EHS):** Environment, Health, Safety, GreenScore  
**Engineered:** log(Time), log(SLR), log(Conc), Conc×Temp, Temp×log(Time), 1/T(K)

---

## 📈 Output Artifacts

| File | Description |
|------|-------------|
| `artifacts/models/preprocessor.joblib` | Fitted sklearn ColumnTransformer |
| `artifacts/models/best_model.joblib`   | Best trained estimator |
| `artifacts/reports/model_comparison.json` | All model metrics |
| `artifacts/reports/evaluation_report.json` | Final R², RMSE, MAE |
| `artifacts/plots/actual_vs_predicted.png` | Scatter plot |
| `artifacts/plots/residuals.png`        | Residual diagnostics |
| `artifacts/plots/shap_summary.png`     | SHAP feature importance |
| `logs/run_YYYYMMDD_HHMMSS.log`        | Full timestamped run log |

---

## 🔭 Next Steps

- [ ] Hyperparameter tuning with `GridSearchCV` / `Optuna`
- [ ] Cross-validation on real data (leave-one-paper-out)
- [ ] Deploy on AWS/GCP with Docker
- [ ] Add multi-target prediction (Co, Li, Ni, Mn simultaneously)
- [ ] Integrate MLflow experiment tracking

---

## 📬 Contact

PhD Research Project — Battery Recycling ML  
Supervisor: [Professor Name], [Department]
