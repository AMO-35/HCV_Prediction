import joblib
import pandas as pd

# Sample input data matching the Hepatitis C dataset
sample_data = {
    'Age': 32,       
    'Sex': 'm',       
    'ALB': 38.5,       
    'ALP': 52.5,       
    'ALT': 7.7,       
    'AST': 22.1,       
    'BIL': 7.5,       
    'CHE': 6.93,       
    'CHOL': 3.23,       
    'CREA': 106.0,       
    'GGT': 12.1,       
    'PROT': 69.0       
}

sample_data_df = pd.DataFrame([sample_data])

# Load the trained model
model = joblib.load('\models\best_model_with_pipeline.pkl')
result = model.predict(sample_data_df)
print(result)