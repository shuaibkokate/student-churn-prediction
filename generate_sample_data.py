import pandas as pd
import numpy as np

np.random.seed(42)

# Total rows
total_rows = 500

# Generate features randomly but realistically
attendance_pct = np.random.uniform(50, 100, total_rows)  # 50% to 100%
avg_grade = np.random.uniform(40, 100, total_rows)      # 40 to 100 marks
engagement_score = np.random.uniform(0, 10, total_rows) # 0 to 10 scale

# Create churn labels:
churn = np.array([1]*100 + [0]*200 + [np.nan]*200)

# Shuffle to mix churn values
indices = np.arange(total_rows)
np.random.shuffle(indices)

attendance_pct = attendance_pct[indices]
avg_grade = avg_grade[indices]
engagement_score = engagement_score[indices]
churn = churn[indices]

# Create DataFrame
df = pd.DataFrame({
    'attendance_pct': attendance_pct,
    'avg_grade': avg_grade,
    'engagement_score': engagement_score,
    'churned': churn
})

df.to_csv("data/student_churn_data.csv", index=False)
print("Sample data saved to data/student_churn_data.csv")
