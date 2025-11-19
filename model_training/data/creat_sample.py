import pandas as pd

print("📌 Loading large dataset...")
df = pd.read_csv("large_data.csv", low_memory=False)

print("📌 Dataset loaded:", df.shape)

print("📌 Sampling 200k rows (random, stratified if possible)...")

# Try to sample equal default/non-default ratio if target exists
if "loan_status" in df.columns:
    # First create default column temporarily
    default_states = [
        "Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)"
    ]
    df["is_default"] = df["loan_status"].isin(default_states).astype(int)

    # Stratified sample — VERY important
    df_sample = df.groupby("is_default", group_keys=False).apply(
        lambda x: x.sample(
            frac=200000/len(df), 
            random_state=42, 
            replace=False
        )
    )
else:
    df_sample = df.sample(n=200000, random_state=42)

print("📌 Sample created:", df_sample.shape)

df_sample.to_csv("sample_200k.csv", index=False)
print("🎉 Saved as sample_200k.csv")
