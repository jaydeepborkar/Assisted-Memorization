"""
quick formatting of data files  
"""

import os
import sys 

def extract_percentage(file_name):
    try:
        base = os.path.splitext(file_name)[0]
        _, percent = base.split('_')
        return int(percent.strip('%'))
    except ValueError:
        return -1  

def clean_data_files(folder_path): 
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    sorted_files = sorted(files, key=extract_percentage)

    for file in sorted_files:
        base, ext = os.path.splitext(file)
        epoch, percent = base.split('_')
        percent = percent.strip('%')

        if 10 <= int(percent) <= 100:
            continue

        if 110 <= int(percent) <= 200:
            new_epoch = 'epoch2'
            new_percent = str(int(percent) - 100)
        else:
            new_epoch = 'epoch3'
            new_percent = str(int(percent) - 200)

        new_filename = f'{new_epoch}_{new_percent}%.txt'

        old_file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(folder_path, new_filename)

        os.rename(old_file_path, new_file_path)
        print(f'Original: {file} -> Renamed: {new_filename}')


def main():
    if len(sys.argv) < 2:
        print("Usage: python process_data_files.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    clean_data_files(folder_path) 

    
if __name__ == "__main__":
    main()
    