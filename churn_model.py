import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(filepath):
    return pd.read_csv(filepath)

def train_churn_model(df):
    X = df[['attendance_pct', 'avg_grade', 'engagement_score']]
    y = df['churned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model Accuracy on test set: {accuracy_score(y_test, y_pred):.2%}")
    return model

def update_predictions(df, model):
    X = df[['attendance_pct', 'avg_grade', 'engagement_score']]
    df['predicted_churn'] = model.predict(X)
    return df

def churn_analysis(df):
    total_students = len(df)
    actual_churn = df['churned'].sum()
    predicted_churn = df['predicted_churn'].sum()
    accuracy = (df['churned'] == df['predicted_churn']).mean()

    print("\n--- Churn Analysis Report ---")
    print(f"Total students: {total_students}")
    print(f"Actual churned students: {actual_churn} ({actual_churn/total_students:.2%})")
    print(f"Predicted churned students: {predicted_churn} ({predicted_churn/total_students:.2%})")
    print(f"Overall prediction accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    data_file = "data/student_churn_data.csv"
    df = load_data(data_file)

    model = train_churn_model(df)

    df = update_predictions(df, model)

    df.to_csv("data/student_churn_data_with_predictions.csv", index=False)

    churn_analysis(df)
