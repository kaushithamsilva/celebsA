import pandas as pd

# Load CSV (replace with your path)
df = pd.read_csv('../data/list_attr_celeba.csv', index_col=0)
df = df.replace(-1, 0)  # Map values

# Count attributes
counts = df.sum(axis=0).sort_values(ascending=False)
print("Samples per attribute:\n", counts)

# For quick stats
print("\nMost frequent:")
print(counts.head(5))
print("\nLeast frequent:")
print(counts.tail(5))
