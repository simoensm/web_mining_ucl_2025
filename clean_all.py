import pandas as pd

df = pd.read_excel('all_products.xlsx')

df_cleaned = df.applymap(lambda s: s.lower() if isinstance(s, str) else s)

df_cleaned.columns = df_cleaned.columns.str.lower()

if 'name' in df_cleaned.columns:
    df_cleaned = df_cleaned.drop_duplicates(subset=['name'], keep='first')
else:
    print("Warning: 'name' column not found. Duplicates were not removed based on name.")

output_file = 'cleaned_all_products.xlsx'
df_cleaned.to_excel(output_file, index=False)

print(f"File saved to: {output_file}")