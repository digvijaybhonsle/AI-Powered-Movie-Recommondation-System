import pickle
import pandas as pd

print("🔄 Starting to resave pickle files...\n")

# 1. Resave df.pkl
print("Resaving df.pkl ...")
df = pd.read_pickle("df.pkl")
df.to_pickle("df.pkl", protocol=4)
print(f"✅ df.pkl resaved | Shape: {df.shape}\n")

# 2. Resave indices.pkl
print("Resaving indices.pkl ...")
with open("indices.pkl", "rb") as f:
    indices = pickle.load(f)
with open("indices.pkl", "wb") as f:
    pickle.dump(indices, f, protocol=4)
print("✅ indices.pkl resaved\n")

# 3. Resave tfidf_matrix.pkl
print("Resaving tfidf_matrix.pkl ...")
with open("tfidf_matrix.pkl", "rb") as f:
    matrix = pickle.load(f)
with open("tfidf_matrix.pkl", "wb") as f:
    pickle.dump(matrix, f, protocol=4)
print(f"✅ tfidf_matrix.pkl resaved | Shape: {getattr(matrix, 'shape', 'N/A')}\n")

# 4. Resave tfidf.pkl
print("Resaving tfidf.pkl ...")
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f, protocol=4)
print("✅ tfidf.pkl resaved\n")

print("🎉 All pickle files have been successfully resaved with protocol=4!")