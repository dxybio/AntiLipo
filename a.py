import pandas as pd

df = pd.read_csv('MD.csv', sep='\t')










# df = df.drop_duplicates(subset=['Name', 'ProteinName'])
# print(df['ProteinName'].value_counts())

df = df.drop_duplicates(subset=['Name'])
print(df['Name'].value_counts())
