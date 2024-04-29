import os
import shutil

def filter_files_by_keyword(source, target, keyword):
    """
    Filters files in the given directory to only include those with the keyword in the filename.

    Parameters:
    - directory: The path to the directory containing the files to be filtered.
    - keyword: The keyword to filter filenames by.

    Returns:
    A list of filenames within the directory that contain the keyword.
    """
    # List all files in the directory
    all_files = os.listdir(source)
    
    # Filter files by keyword
    for file in all_files:
        if keyword.lower() in file.lower():
            shutil.copy(os.path.join(source, file), target_path)
        
    
 

directory_path = '/home/lihanzhao/Documents/SparceGNN4Brain/HCP_Data/structural_network_132/'  # Replace with your directory path
target_path = '/home/lihanzhao/Documents/SparceGNN4Brain/HCP_Data/structural_network_132_0/'
keyword = 'density'
filtered_files = filter_files_by_keyword(directory_path, target_path, keyword)

all_files = os.listdir(target_path)
i = 0
for file in all_files:
    i += 1
print(i)