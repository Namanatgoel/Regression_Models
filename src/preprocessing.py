import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_linear_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Preprocesses data for the Linear Regression model (Boston Housing).
    Implements median imputation for 'CRIM' and scaling without inplace warnings.
    """
    df_processed = df.copy()
    
    if 'CRIM' in df_processed.columns:
        df_processed['CRIM'] = df_processed['CRIM'].fillna(df_processed['CRIM'].median())
    
    for col in df_processed.columns:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())

    feature_names = df_processed.drop(columns=['MEDV']).columns.tolist()
    X = df_processed.drop(columns=['MEDV']).values
    y = df_processed['MEDV'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_names

def preprocess_logistic_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses data for Logistic Regression model (Titanic).
    Drops uninformative string columns, handles missing values, and one-hot encodes categorical features.
    """
    df_processed = df.copy()

    # Drop categorical/uninformative columns that cannot be cast to float
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df_processed = df_processed.drop(columns=[c for c in cols_to_drop if c in df_processed.columns])

    if 'Age' in df_processed.columns:
        df_processed['Age'] = df_processed['Age'].fillna(df_processed['Age'].median())

    if 'Embarked' in df_processed.columns:
        df_processed['Embarked'] = df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0])

    if 'Survived' in df_processed.columns:
        df_processed = df_processed.dropna(subset=['Survived'])
        y = df_processed['Survived'].values.astype(float)
        X_df = df_processed.drop(columns=['Survived'])
    else:
        raise KeyError("Target column 'Survived' not found in dataset.")

    # One-Hot Encoding for remaining categorical string features ('Sex', 'Embarked')
    X_df = pd.get_dummies(X_df, columns=[c for c in ['Sex', 'Embarked'] if c in X_df.columns], drop_first=True)

    X = X_df.values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y