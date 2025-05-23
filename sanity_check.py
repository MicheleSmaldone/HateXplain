import json
with open("Data/dataset.json") as f:
    data = json.load(f)
print("Loaded", type(data), "with", len(data), "entries.")
if isinstance(data, dict):
    print("First few post IDs:", list(data)[:5])
else:
    print("Sample element:", data[0])
