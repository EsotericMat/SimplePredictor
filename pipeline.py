import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    confusion_matrix
    )


def get_data(path):
    df = pd.read_csv(path)
    X = df.drop('label', axis=1)
    y = df['label']
    return X, y


def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)


def train_model(X_train, y_train):
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, f1, precision, cm


def evaluation_report(accuracy, f1, precision, cm):
    print("Accuracy:", accuracy)
    print("F1 score:", f1)
    print("Precision:", precision)
    print("Confusion matrix:\n", cm)
    return None


def save_model(model):
    print("Saving model...")
    joblib.dump(model, 'classifier.joblib')
    print("✅ Model saved as 'classifier.joblib'")


if __name__ == '__main__':
    X, y = get_data('datasets/blobs.csv')
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    a, b, c, d = evaluate_model(model, X_test, y_test)
    evaluation_report(
        a, b, c, d
    )
    save_model(model)

