import joblib

features = joblib.load("updated_features.pkl")

print("Type of object:", type(features))
print("\nFeatures inside:\n")
print(features)