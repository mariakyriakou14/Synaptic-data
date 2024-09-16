import os
import pandas as pd
import sys
import re  # Regular expression module for pattern matching

def read_data(folder_paths: list[str]) -> pd.DataFrame:
    # Define regions by image numbers and categories
    regions = {
        'EC-Lyll': ['Image_10', 'Image_11', 'Image_12'],
        'Ca3-Rad': ['Image_4', 'Image_5', 'Image_6'],
        'Ca1-LMol': ['Image_7', 'Image_8', 'Image_9'],
        'PoDG': ['Image_1', 'Image_2', 'Image_3']
    }
    categories = ["MAPT_KI_WT", "P301S", "S305N"]

    all_data = []  # List to collect all data frames

    # Iterate over all folders provided as input
    for folder_path in folder_paths:
        # List all csv files in the current directory
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv') and not any(tag in filename for tag in ["Overall", "Time"]):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path, skiprows=3)
                print(f"Processing {file_path}")
                # Extract statistic name from the filename
                start = filename.find('parameters_') + len('parameters_')
                end = filename.rfind('.csv')
                statistic_name = df.columns[0]

                # Extract image tag and determine the region
                image_tag = re.search(r'Image_(\d+)', filename)
                if image_tag:
                    image_number = image_tag.group(0)
                    region = next((r for r, tags in regions.items() if image_number in tags), None)
                    if region:
                        # Populate DataFrame with required details
                        df['Region'] = region
                        df['Statistic'] = statistic_name
                        
                        sample_end_index = filename.index(f'_{image_number}')  # Find where the sample name ends
                        df['Sample'] = filename[:sample_end_index]
                        
                        df['Category'] = next((cat for cat in categories if cat in filename), 'Unknown Category')
                        all_data.append(df)

    # Combine all individual data frames into a single one
    return pd.concat(all_data, ignore_index=True)

def process_and_save_data(full_data: pd.DataFrame, regions:dict[str, list[str]]):
    # Calculate means and reshape data for each region and statistic
    for region in regions:
        region_data = full_data[full_data['Region'] == region]
        statistics = region_data['Statistic'].unique()
        for stat in statistics:
            means = region_data[region_data['Statistic'] == stat].groupby(['Category', 'Sample', 'Surpass Object']).agg({stat: 'mean'}).reset_index()
            pivot_df = means.pivot_table(index=['Category', 'Sample'], columns='Surpass Object', values=stat, aggfunc='first').reset_index()
            pivot_df.columns = [f"{col[1]}Average{col[0]}" if col[1] else col[0] for col in pivot_df.columns]

            # Save the results to CSV
            output_path = f"{region}_{stat}.csv"
            pivot_df.to_csv(output_path, index=False)
            print(f"Saved data for {region} {stat} to {output_path}")

if __name__ == "__main__":
    # Accept multiple folder paths from command line
    folders = sys.argv[1:]
    full_data = read_data(folders)
    full_data.to_csv("all_data.csv")
    regions_keys = {
        'EC-Lyll': ['Image_10', 'Image_11', 'Image_12'],
        'Ca3-Rad': ['Image_4', 'Image_5', 'Image_6'],
        'Ca1-LMol': ['Image_7', 'Image_8', 'Image_9'],
        'PoDG': ['Image_1', 'Image_2', 'Image_3']
    }
    process_and_save_data(full_data, regions_keys)
