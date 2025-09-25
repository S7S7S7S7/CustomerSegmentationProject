import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess dataset for clustering:
    - Drop duplicates
    - Drop datetime columns
    - Fill missing values
    - Encode categorical columns
    - Scale numeric features
    """

    # 1. Drop duplicates
    df = df.drop_duplicates()

    # 2. Drop datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns
    df = df.drop(columns=datetime_cols)

    # 3. Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        elif np.issubdtype(df[col].dtype, np.number):
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(0, inplace=True)

     # Encode categorical features
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Select numeric columns for scaling
    numeric_cols = df.select_dtypes(include='number').columns
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Only keep numeric columns for clustering
    df_for_model = df[numeric_cols]

    return df, scaler, label_encoders
