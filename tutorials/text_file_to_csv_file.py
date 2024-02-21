from pathlib import Path
import numpy as np
import csv

def txt_file_to_csv(folder_path: Path, file_name: str) -> Path:
    """Given a folder path (for example path to 'pillar_results' folder) that contains a '.txt' file and the file name 
    of the text file (for example 'pillar1_col'). Will convert the text file into a '.csv' file and save it as 
    'file_name.csv' in the same folder path."""
    file_path = str(folder_path) + '/' + file_name + '.txt'
    output_path = str(folder_path) + '/' + file_name + '.csv'
    text_as_array = np.loadtxt(file_path) 
    with open(output_path, 'w') as out_file:
        writer = csv.writer(out_file)
        if len(text_as_array.shape) < 2:
            writer.writerows(zip(text_as_array))
        else:
            writer.writerows(text_as_array)
    return Path(output_path)