import joblib
m = joblib.load("/home/bura/projects/final_year/deployment_models/delhi/lightgbm.joblib")
print("Features:", m["features"])
print("Target:", m["target"])
