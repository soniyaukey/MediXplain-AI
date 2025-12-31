import pandas as pd
import numpy as np
import os

def load_and_preprocess_data():
    """
    Load and preprocess the PIMA Diabetes Dataset.
    """
    # Define column names based on the dataset description
    column_names = [
        'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
        'insulin', 'bmi', 'diabetes_pedigree', 'age', 'outcome'
    ]

    # Load the data
    data_path = 'data/pima-indians-diabetes.data'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}. Please run download.py first.")

    df = pd.read_csv(data_path, header=None, names=column_names)

    # Handle missing values (represented as 0 in this dataset for certain features)
    zero_features = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
    for feature in zero_features:
        df[feature] = df[feature].replace(0, df[feature].median())

    # Separate features and target
    X = df.drop('outcome', axis=1)
    y = df['outcome']

    feature_names = list(X.columns)

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Feature names: {feature_names}")

    return X.values, y.values, feature_names

if __name__ == "__main__":
    load_and_preprocess_data()
