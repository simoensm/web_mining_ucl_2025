import pandas as pd
import glob
import os

FILE_TYPE = 'xlsx'  
OUTPUT_FILENAME = 'all_products.xlsx'
# ---------------------

def merge_files():
    search_pattern = f"*.{FILE_TYPE}"
    files = glob.glob(search_pattern)
    
    if OUTPUT_FILENAME in files:
        files.remove(OUTPUT_FILENAME)
        
    all_data_frames = []
    
    print(f"Found {len(files)} files to merge...")
    
    for filename in files:
        try:
            if FILE_TYPE == 'xlsx':
                df = pd.read_excel(filename)
            else:
                df = pd.read_csv(filename)
            
            df['Source_File'] = filename
            
            all_data_frames.append(df)
            print(f" -> Successfully read: {filename} ({len(df)} rows)")
            
        except Exception as e:
            print(f" ! Error reading {filename}: {e}")

    if all_data_frames:
        print("Merging data...")
        merged_df = pd.concat(all_data_frames, ignore_index=True)
   
        merged_df.to_excel(OUTPUT_FILENAME, index=False)
        print(f"\nSUCCESS: All files merged into '{OUTPUT_FILENAME}'")
        print(f"Total products: {len(merged_df)}")
    else:
        print("No files found to merge.")

if __name__ == "__main__":
    merge_files()