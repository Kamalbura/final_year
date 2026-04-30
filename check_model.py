import joblib
m = joblib.load("/home/bura/projects/final_year/deployment_models/delhi/lightgbm.joblib")
print(f"Type: {type(m)}")
if isinstance(m, dict):
    print(f"Keys: {m.keys()}")
    if 'model' in m:
        print(f"Model type: {type(m['model'])}")
else:
    print("Direct model object")
