import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("final_dataset.csv")

# Encoding maps
cgpa_map = {
    "Less than 6.5": 0,
    "6.5 to 8.5": 1,
    "More than 8.5": 2
}

cc_map = {
    "Every week": 4,
    "Monthly": 3,
    "Only during exams": 2,
    "When internet is not available at hostels": 1,
    "Never": 0
}

lib_map = {
    "Every week": 3,
    "Monthly": 2,
    "Only during exams": 1,
    "Never": 0
}

att_map = {
    "<50%": 0,
    "50-75%": 1,
    "75-90%": 2,
    "90-99%": 3,
    "I never miss classes unless there is an emergency": 4
}

backlog_map = {"No": 0, "Yes": 1}

res_map = {
    "Library resources (books, journals)": 0,
    "Faculty office hours/support": 1,
    "Tutoring/Academic support services/YouTube": 2,
    "Seniors": 3
}

# Apply encoding
df["y"] = df["How much is your CGPA"].map(cgpa_map)
df["cc"] = df["How much do you use the college computer center (CC) for studying?"].map(cc_map)
df["lib"] = df["How much do you use the college library for studying?"].map(lib_map)
df["res"] = df["Which of the following resources provided by the college do you find most beneficial for academic success? (Select all that apply)"].map(res_map)
df["adv"] = df["On a scale of 1 to 5, how would you rate the effectiveness of the college's academic advising services?"]
df["study"] = df["How many hours per week, on average, do you spend on activities directly related to your studies (e.g., homework, self-study)?"]
df["att"] = df["How much attendance do you manage in a semester?"].map(att_map)
df["back"] = df["Did you have any previous backlogs?"].map(backlog_map)

# Features
X = df[["cc","lib","res","adv","study","att","back"]]
y = df["y"]

# Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)

# Save
joblib.dump(model, "model/cgpa_model.joblib")
