# SickleSense
Sickle cell early warning signs, detection and care
This project proposes a digital health solution designed to predict pain crises for sickle cell patients, provide personalized preventative guidance, and improve outcomes in both wealthy and underserved African communities.

# Problem Definition
I set out to build a machine learning model that predicts pain intensity & type in sickle cell patients using clinical data, environmental factors, activity signals, and patient-reported symptoms. The goal is to support early detection of high-risk pain episodes and improve patient self-management.

# Dataset Creation
I used a Kaggle sickle cell dataset as the foundation for this project. I cleaned and prepared the data by handling missing values, fixing inconsistent formats, encoding categorical features, and normalizing numerical fields. I also engineered additional features to better support pain prediction. The final curated dataset became the input for the model training pipeline.
- Demographics
- Clinical vitals (HR, SpO₂, BP, respiratory rate, Hb, reticulocyte count)
- Patient-reported symptoms
- Weather/environment variables
- Activity/wearable-like metrics
- Time-based features
- Pain intensity (target variable)
- Pain type (target variable 2)

# Model Building & Training
- Selected a supervised machine learning approach to predict pain intensity.
- Split the dataset into training and testing sets to ensure proper evaluation.
- Trained multiple models (e.g., Random Forest, Logistic Regression, Gradient Boosting) to compare performance.
- Tuned hyperparameters to improve accuracy, reduce overfitting, and stabilize predictions.
- Chose the best-performing model based on real-world reliability and consistency.
- Saved the trained model for deployment and integration into the app environment.

# Key Insights
Features that contributed significantly to higher predicted pain scores included:
- Past pain score
- Oxygen saturation
- Heart rate
- Fatigue level
- Hemoglobin
Environmental variables like temperature changes and humidity also showed moderate influence.

# Next Steps
- Increase dataset size or integrate real-world SCD datasets
- Add time-series modeling (LSTM) for continuous monitoring
- Build a simple UI to visualize patient metrics and predictions(Done)
- Deploy model as an API endpoint for a mobile app prototype(Done)

# Summary
This commit includes:
- Synthetic dataset (sickle_cell_clinical_note.csv)
- Data preprocessing script
- Model training notebook
- Evaluation metrics and feature importance analysis
- Notes on assumptions and clinical considerations
