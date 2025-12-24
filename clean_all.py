import pandas as pd

# 1. Load the Excel file
# Ensure 'all_products.xlsx' is in the same folder as your script
df = pd.read_excel('all_products.xlsx')

# 2. Lowercase every text cell in the dataframe
df_cleaned = df.applymap(lambda s: s.lower() if isinstance(s, str) else s)

# Optional: Lowercase column headers as well to ensure 'Name' becomes 'name'
df_cleaned.columns = df_cleaned.columns.str.lower()

# 3. Remove duplicates based ONLY on the 'name' column
# This keeps the first occurrence and removes subsequent ones
if 'name' in df_cleaned.columns:
    df_cleaned = df_cleaned.drop_duplicates(subset=['name'], keep='first')
else:
    print("Warning: 'name' column not found. Duplicates were not removed based on name.")

# 4. Save the result to a new Excel file
output_file = 'cleaned_all_products.xlsx'
df_cleaned.to_excel(output_file, index=False)

print(f"File saved to: {output_file}")