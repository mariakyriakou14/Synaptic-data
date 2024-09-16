import os
import sys

def find_matching_folders(base_path, prefix, suffix):
    """
    Finds and prints the paths of all folders under base_path matching the specified prefix and suffix patterns.
    
    Parameters:
        base_path (str): The directory path where the search will begin.
        prefix (str): The prefix pattern for folder names.
        suffix (str): The suffix pattern for folder names.
    """
    matching_folders = []

    # Debugging: print the parameters to ensure they are passed correctly
    print(f"Base path: {base_path}")
    print(f"Prefix: {prefix}")
    print(f"Suffix: {suffix}")

    # Walk through the directory
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            # Debugging: show which directories are being checked
            # print(f"Checking directory: {os.path.join(root, dir_name)}")
            
            if dir_name.startswith(prefix) and dir_name.endswith(suffix):
                full_path = os.path.join(root, dir_name)
                matching_folders.append(full_path)
                # print(f"Match found: {full_path}")  # Show matched directoriespyto
                
    output = "' '".join(matching_folders)
    print(output)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python path_builder.py <base_directory> <prefix> <suffix>")
        sys.exit(1)

    base_directory = sys.argv[1]
    prefix = sys.argv[2]
    suffix = sys.argv[3]
    
    find_matching_folders(base_directory, prefix, suffix)
