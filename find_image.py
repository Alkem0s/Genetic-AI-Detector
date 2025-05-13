import os
import pandas as pd

def find_rows_for_jpgs(csv_path, directory_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Extract base file names (e.g., "abc.jpg") from full paths in the CSV
    df['base_name'] = df['file_name'].apply(lambda x: os.path.basename(x))
    
    # List all .jpg files in the given directory
    jpg_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.jpg')]
    
    # Filter DataFrame to rows that match the .jpg files
    matching_rows = df[df['base_name'].isin(jpg_files)]
    
    return matching_rows

csv_path = 'train.csv'
images_dir = os.getcwd()
matches = find_rows_for_jpgs(csv_path, images_dir)
print(matches)
