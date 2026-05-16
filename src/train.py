import argparse
from src.data_loader import load_data
from src.preprocessing import preprocess_linear_data, preprocess_logistic_data
from src.models.linear_model import VectorizedLinearRegression
from src.models.logistic_model import VectorizedLogisticRegression
from src.evaluate import evaluate_linear, evaluate_logistic
import numpy as np
import logging

from src.evaluate import evaluate_linear, evaluate_logistic, plot_correlation_heatmap
from src.data_loader import load_data
from src.preprocessing import preprocess_linear_data, preprocess_logistic_data

def train_linear(data_path: str):
    df = load_data(data_path)
    X, y, feature_names = preprocess_linear_data(df)
    
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = VectorizedLinearRegression(learning_rate=0.01, epochs=2000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    evaluate_linear(y_test, y_pred, feature_names, model.weights)

def train_logistic(data_path: str):
    df = load_data(data_path)
    X, y = preprocess_logistic_data(df)

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = VectorizedLogisticRegression(learning_rate=0.01, epochs=2000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    evaluate_logistic(y_test, y_pred, y_prob)

def train_linear(data_path: str):
    df = load_data(data_path)
    
    plot_correlation_heatmap(df, "Linear")
    
    X, y, feature_names = preprocess_linear_data(df)
    
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = VectorizedLinearRegression(learning_rate=0.01, epochs=2000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    evaluate_linear(y_test, y_pred, feature_names, model.weights)

def train_logistic(data_path: str):
    df = load_data(data_path)
    
    plot_correlation_heatmap(df, "Logistic")
    
    X, y = preprocess_logistic_data(df)
    
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = VectorizedLogisticRegression(learning_rate=0.05, epochs=5000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) 

    evaluate_logistic(y_test, y_pred, y_prob)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Models")
    parser.add_argument("--model", type=str, required=True, choices=["linear", "logistic"], help="Model type to train")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    args = parser.parse_args()

    if args.model == "linear":
        train_linear(args.data)
    elif args.model == "logistic":
        train_logistic(args.data)