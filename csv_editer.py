import pandas as pd

file = "master_pin_full.csv"

df = pd.read_csv(file)
df = df[df["League"] != "Not Found"]
df.to_csv(file,index=False)



