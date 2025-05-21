import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

def load_data(filepath):
    return pd.read_csv(filepath)

def balance_data(df):
    df = df.dropna(subset=['churned'])
    df_majority = df[df.churned == 0]
    df_minority = df[df.churned == 1]

    df_minority_upsampled = resample(df_minority, 
                                     replace=True, 
                                     n_samples=len(df_majority), 
                                     random_state=42)

    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    return df_balanced

def train_churn_model(df):
    X = df[['attendance_pct', 'avg_grade', 'engagement_score']]
    y = df['churned'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n--- Model Evaluation ---")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model

def update_predictions(df, model):
    X = df[['attendance_pct', 'avg_grade', 'engagement_score']]
    df['predicted_churn'] = model.predict(X)
    return df

def churn_analysis(df):
    df_clean = df.dropna(subset=['churned', 'predicted_churn'])
    
    total_students = len(df_clean)
    actual_churn = df_clean['churned'].sum()
    predicted_churn = df_clean['predicted_churn'].sum()
    accuracy = (df_clean['churned'] == df_clean['predicted_churn']).mean()

    print("\n--- Churn Analysis Report ---")
    print(f"Total students: {total_students}")
    print(f"Actual churned students: {actual_churn} ({actual_churn/total_students:.2%})")
    print(f"Predicted churned students: {predicted_churn} ({predicted_churn/total_students:.2%})")
    print(f"Overall prediction accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    data_file = "data/student_churn_data.csv"
    df = load_data(data_file)

    # Balance and train only on labeled data
    labeled_df = df.dropna(subset=['churned'])
    balanced_df = balance_data(labeled_df)
    model = train_churn_model(balanced_df)

    # Predict churn for all students (including unlabeled)
    df = update_predictions(df, model)

    # Save updated data with predictions
    df.to_csv("data/student_churn_data_with_predictions.csv", index=False)

    # Generate churn analysis only on labeled data
    churn_analysis(df)
