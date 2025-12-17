import numpy as np

path = "./test_data/001.npy"
data = np.load(path, allow_pickle=True)

print("Type:", type(data))
print("Shape:", getattr(data, "shape", None))
print("Dtype:", data.dtype)
