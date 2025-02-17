"""
quick formatting of checkpoints dirs 
""" 

import os
import sys 


def get_numerical_part(folder_name):
    try:
        return int(folder_name.split('-')[1])
    except (IndexError, ValueError):
        return float('inf')  
    
def clean_checkpoints(folder_path): 
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    sorted_folders = sorted(folders, key=get_numerical_part)

    for i, folder in enumerate(sorted_folders):
        if i < 10:
            new_name = f'{(i + 1) * 10}e1'
        elif i < 20:
            new_name = f'{(i - 9) * 10}e2'
        else:
            new_name = f'{(i - 19) * 10}e3'
    
        old_folder_path = os.path.join(folder_path, folder)
        new_folder_path = os.path.join(folder_path, new_name)
    
        os.rename(old_folder_path, new_folder_path)
        print(f'Original: {folder} -> Renamed: {new_name}')
        
    
def main():
    if len(sys.argv) < 2:
        print("Usage: python process_checkpoints.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    clean_checkpoints(folder_path) 

    
if __name__ == "__main__":
    main()
