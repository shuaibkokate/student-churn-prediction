import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(filepath):
    return pd.read_csv(filepath)

def train_churn_model(df):
    # Drop rows with missing features (but allow missing churn labels)
    df = df.dropna(subset=['attendance_pct', 'avg_grade', 'engagement_score'])

    # Only use rows with churn labels for training
    train_df = df.dropna(subset=['churned'])
    X = train_df[['attendance_pct', 'avg_grade', 'engagement_score']]
    y = train_df['churned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model trained. Validation Accuracy: {acc:.2%}")

    return model

def update_predictions(df, model):
    # Predict churn only for rows where churned is missing
    prediction_df = df[df['churned'].isna()].copy()
    X_pred = prediction_df[['attendance_pct', 'avg_grade', 'engagement_score']]
    prediction_df['predicted_churn'] = model.predict(X_pred)

    # Merge predictions back into original dataframe
    df.update(prediction_df)
    return df

def churn_analysis(df):
    total_students = len(df)
    predicted_churn = df['predicted_churn'].sum()

    print("\n--- Churn Prediction Summary ---")
    print(f"Total students: {total_students}")
    print(f"Predicted churned students: {predicted_churn} ({predicted_churn / total_students:.2%})")

if __name__ == "__main__":
    data_file = "data/student_churn_data.csv"
    df = load_data(data_file)

    model = train_churn_model(df)

    # Predict churn for unlabeled data
    df = update_predictions(df, model)

    # Save results
    df.to_csv("data/student_churn_data_with_predictions.csv", index=False)

    # Analysis
    churn_analysis(df)
