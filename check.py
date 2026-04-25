import pandas as pd

df = pd.read_csv('data/raw/HI-Small_Trans.csv')
laundering = df[df['Is Laundering'] == 1]
print("Laundering transactions:", len(laundering))
print("Unique sender accounts:", laundering['Account'].nunique())
print("Unique receiver accounts:", laundering['Account.1'].nunique())
print("Total unique accounts involved:", 
      pd.concat([laundering['Account'], 
                laundering['Account.1']]).nunique())