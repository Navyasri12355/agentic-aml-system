import pandas as pd

# Load files
print("Loading files...")
flagged = pd.read_csv('data/processed/flagged_hybrid_final.csv')
raw = pd.read_csv('data/raw/HI-Small_Trans.csv')

# Get all unique laundering accounts from raw
laundering_rows = raw[raw['Is Laundering'] == 1]
laundering_accts = set(
    laundering_rows['Account'].astype(str).tolist() +
    laundering_rows['Account.1'].astype(str).tolist()
)

# Get all unique accounts in flagged transactions
flagged_accts = set(
    flagged['sender_id'].astype(str).tolist() +
    flagged['receiver_id'].astype(str).tolist()
)

# Find overlap
overlap = laundering_accts.intersection(flagged_accts)

print(f"Total laundering accounts in raw data : {len(laundering_accts)}")
print(f"Total unique accounts in flagged      : {len(flagged_accts)}")
print(f"Laundering accounts in flagged        : {len(overlap)}")
print(f"Coverage                              : {len(overlap)/len(laundering_accts):.2%}")
print(f"Laundering accounts NOT in flagged    : {len(laundering_accts) - len(overlap)}")