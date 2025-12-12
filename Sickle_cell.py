import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

df = pd.read_csv("sickle_cell_clinical_note.csv")
df = df.drop([
    "patient_id",
    "visit_datetime",
    "facility_type",
    "location",
    "clinical_note",
    "diagnosis_date",
    "race"
], axis=1)

df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
df["age"] = (pd.Timestamp.today() - df["dob"]).dt.days // 365
df = df.drop("dob", axis=1)

y_class = df["pain_type"]  # classification target
y_reg = df["pain_intensity"]  # regression target

X = df.drop(["pain_type", "pain_intensity"], axis=1)

categorical = ["admitted", "weather", "gender"]
numeric = ["hemoglobin", "oxygen_saturation", "age"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numeric)
])

# Pain type model
clf_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    ))
])

# Pain intensity model
reg_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("reg", RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42
    ))
])

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

clf_model.fit(X_train_clf, y_train_clf)
reg_model.fit(X_train_reg, y_train_reg)


#print(clf_model.predict([[20,3,10,5,3,7]]))
def predict_pain_type(admitted, weather, gender, hemoglobin, oxygen, age):
    new_patient = pd.DataFrame([{
        "admitted": admitted,
        "weather": weather,
        "gender": gender,
        "hemoglobin": hemoglobin,
        "oxygen_saturation": oxygen,
        "age": age
    }])
    return clf_model.predict(new_patient)[0]


def predict_pain_intensity(admitted, weather, gender, hemoglobin, oxygen, age):
    new_patient = pd.DataFrame([{
        "admitted": admitted,
        "weather": weather,
        "gender": gender,
        "hemoglobin": hemoglobin,
        "oxygen_saturation": oxygen,
        "age": age
    }])
    return reg_model.predict(new_patient)[0]

example_type = predict_pain_type(
    #admitted="yes",
    admitted = str(input("Have you been admitted before? (Y/N): ")),
    #weather="cold",
    weather = str(input("What is the weather? (C/H): ")),
    #gender="male",
    gender = str(input("What is your gender? (M/F): ")),
    #hemoglobin=5.9,
    hemoglobin = float(input("What is your hemoglobin?: ")),
    #oxygen=75,
    oxygen= int(input("What is your oxygen level?: ")),
    #age=37
    age = int(input("What is your age?: ")
    )
)

example_intensity = predict_pain_intensity(
    #admitted="yes",
    admitted = str(input("Have you been admitted before? (Y/N): ")),
    #weather="cold",
    weather = str(input("What is the weather? (C/H): ")),
    #gender="male",
    gender = str(input("What is your gender? (M/F): ")),
    #hemoglobin=5.9,
    hemoglobin = float(input("What is your hemoglobin?: ")),
    #oxygen=75,
    oxygen= int(input("What is your oxygen level?: ")),
    #age=37
    age = int(input("What is your age?: "))
)

print("Predicted Pain Type:", example_type)
print("Predicted Pain Intensity:", example_intensity)


fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.scatter( example_type, example_intensity, marker='x', c='red')
ax.set_xlabel("Age")
ax.set_ylabel("Pain Intensity")
plt.title("Pain Intensity vs Age Categorized by Pain Type")
plt.legend(title="Pain Type")
plt.grid(True)

plt.show()
import joblib
joblib.dump(clf_model, 'clf_model.joblib')
joblib.dump(reg_model, 'reg_model.joblib')

import pickle
with open('classifier.pkl', "wb") as file:
    pickle.dump(clf_model, file)
with open('regressor.pkl', "wb") as file:
    pickle.dump(reg_model, file)
