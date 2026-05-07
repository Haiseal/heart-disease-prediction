<<<<<<< HEAD
# Heart Disease Prediction using Big Data & Machine Learning

## BRFSS 2022 | XGBoost + SMOTE | Hadoop HDFS + Apache Spark

### Project Overview
End-to-end Big Data ML pipeline for CVD risk prediction.

- **Dataset:** BRFSS 2022 (340,154 records, 50 features)
- **Model:** XGBoost + SMOTE + Optuna tuning
- **AUC:** 0.830 | Sensitivity: 0.786 | G-mean: 0.755
- **Stack:** Hadoop 3.3.6 + Spark 3.5.1 + Python 3.10

### Pipeline
### Notebooks
| Notebook | Description |
|---|---|
| NB1 | Data Preprocessing |
| NB2 | Exploratory Data Analysis |
| NB3A | Baseline Models (LR, RF, XGBoost, NB) |
| NB3B | Optuna Tuning + SHAP |
| NB4 | Model Export |

### Web App
```bash
cd app/
pip install -r requirements.txt
streamlit run app.py
```

### Dataset
Download BRFSS 2022 from: https://www.cdc.gov/brfss/annual_data/annual_2022.html
=======
# heart-disease-prediction
CVD Risk Prediction using BRFSS 2022 + Big Data Pipeline
>>>>>>> 4a1d263cc4b979d82dc5c24c98b7cd71fa9f0f0a
