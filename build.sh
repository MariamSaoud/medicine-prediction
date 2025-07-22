# download_model.py
import os, requests

url = "https://github.com/MariamSaoud/medicine-prediction/raw/main/knn_model.pkl"
dest = "knn_model.pkl"

if not os.path.exists(dest):
    print("📥 Downloading model...")
    response = requests.get(url)
    with open(dest, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded.")
else:
    print("🟢 Model already exists.")