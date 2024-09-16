import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def load_data(folders: list[str]):
    """
    Load and process data from specified folders and regions.

    Parameters:
        folders (list[str]): List of folder paths containing the data.
        regions (dict): Dictionary mapping region names to image indices.

    Returns:
        pd.DataFrame: Processed data containing synapse counts and other metrics.
    """
    all_data = pd.DataFrame()
    for folder in folders:
        for file in os.listdir(folder):
            region = "Image_4" in file  or "Image_5"  in file or "Image_6" in file
            if not region:
                continue
            region = file
            data = pd.read_csv(os.path.join(folder, file), header=2)
            spots=0
            for synapse_type in ['Homer1 colocated', 'Synaptotagmin colocated']:
                synapse_data = data[(data['Variable'] == 'Number of Spots per Time Point') & (data['Surpass Object'] == synapse_type)]
                print(file)
                spots = spots + int(synapse_data['Value'])
            category = file.split('_')[0]  # Assuming file names contain genotype as the first part
            if category == "MAPT":
                category = "MAPT KI WT"
            #  Create a new DataFrame for the current result
            avg_spots = spots/ 2 / 7269  # Normalize by image volume or area
            new_data = pd.DataFrame({'Region': [region], 'Category': [category], 'AverageSpots': [avg_spots]})

            # Concatenate the new data with the all_data DataFrame
            all_data = pd.concat([all_data, new_data], ignore_index=True)
            # all_data = all_data.append({'Region': region, 'Category': category, 'SynapseType': synapse_type, 'AverageSpots': avg_spots}, ignore_index=True)

    return all_data

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the dataframe by removing _ and groups by category and then Sample

    Args:
        df (pd.DataFrame): the dataframe with format {'Region': [region], 'Category': [category], 'AverageSpots': [avg_spots]}

    Returns:
        _type_:The grouped dataframe
    """
    def extract_sample(region):
        # If 'mapt' is in the region, split and take the first 5 parts
        if 'mapt' in region.lower():
            return '_'.join(region.split('_')[:5])
        # Otherwise, take the first 3 parts
        else:
            return '_'.join(region.split('_')[:3])
    # Extracting 'Sample' from 'Region'
    df['Sample'] = df['Region'].apply(extract_sample)
    
    # Grouping by 'Category' and 'Sample' and computing mean of 'AverageSpots'
    df_mean = df.groupby(['Category', 'Sample'])['AverageSpots'].mean().reset_index()
    return df_mean

def plot_box(df: pd.DataFrame, output_dir: str):
    """
    Plot box plots for each numeric attribute in the DataFrame with customized colors, 
    and annotate each plot with the median value.

    Parameters:
        df (pd.DataFrame): The DataFrame to plot KDEs for.
        output_dir (str): The directory where the plots will be saved.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        
        plt.figure(figsize=(10, 6))  # Set the size of the plot

        # Create the box plot with customized colors
        ax = sns.boxplot(data=df, x='Category', y=col, palette={'MAPT KI WT': 'green', 'P301S': 'red', 'S305N': 'blue'},order=['MAPT KI WT', 'P301S', 'S305N'])
        

        # Customizing plot with titles and labels
        plt.title(f'Distribution of {col} Across Groups')  # Set title to reflect the column name
        plt.xlabel('Group Type')  # Label for x-axis

        # Annotate median values on the plot
        medians = df.groupby(['Category'])[col].median().values
        median_labels = [str(np.round(s, 2)) for s in medians]
        pos = range(len(medians))
        for tick, label in zip(pos, ax.get_xticklabels()):
            ax.text(pos[tick], medians[tick], median_labels[tick], 
                    horizontalalignment='center', size='x-small', color='white', weight='semibold')
            

        # Save the plot
        plot_path = os.path.join(output_dir, f'plot_{col}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'Plot saved to {plot_path}')

def plot_box(df: pd.DataFrame, output_dir: str):
    """
    Plot box plots for each numeric attribute in the DataFrame with customized colors, 
    and annotate each plot with the median value.

    Parameters:
        df (pd.DataFrame): The DataFrame to plot box plots for.
        output_dir (str): The directory where the plots will be saved.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Plotting
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df, x='Sample', y='AverageSpots', hue='Category',
                     palette={'MAPT KI WT': 'green', 'P301S': 'red', 'S305N': 'blue'})
    
    # Customizing plot with titles and labels
    plt.title('Distribution of Average Spots Across Samples and Categories')
    plt.xlabel('Sample')
    plt.ylabel('Average Spots')
    
    # Annotate median values on the plot
    medians = df.groupby(['Sample', 'Category'])['AverageSpots'].median().values
    median_labels = [f'{np.round(s, 2)}' for s in medians]
    pos = range(len(medians))
    for tick, label in zip(pos, ax.get_xticklabels()):
        ax.text(pos[tick], medians[tick], median_labels[tick],
                horizontalalignment='center', size='x-small', color='white', weight='semibold')

    # Save the plot
    plot_path = os.path.join(output_dir, 'boxplot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Plot saved to {plot_path}')

def plot_interactive_histograms(df: pd.DataFrame, output_dir: str):
    """
    Plot interactive histograms for each numeric attribute in the DataFrame using Plotly.

    Parameters:
        df (pd.DataFrame): The DataFrame to plot histograms for.
        output_dir (str): The directory where the plots will be saved as HTML files.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        fig = px.histogram(df, x=col, color="Category", marginal="box",  # marginal type could be 'box', 'violin', etc.
                        title=f'Distribution of {col} Across Groups',
                        labels={"Group": "Group Type"},color_discrete_map={'MAPT KI WT': 'green', 'P301S': 'red', 'S305N': 'blue'},category_orders={"Category":['MAPT KI WT', 'P301S', 'S305N'].__reversed__()})  # Label for legend

        # Save the plot as an HTML file which can be opened in a browser for interactive use
        plot_path = os.path.join(output_dir, f'{col}_interactive.html')
        fig.write_html(plot_path)
        print(f'Interactive plot saved to {plot_path}')

def plot_histograms(df: pd.DataFrame, output_dir: str):
    numeric_cols = df.select_dtypes(include=['number']).columns
    color_map = {'MAPT KI WT': 'green', 'P301S': 'red', 'S305N': 'blue'}  # Define color map

    for col in numeric_cols:
        plt.figure(figsize=(10, 6))  # Set the size of the plot

        # Calculate means for each category
        means = df.groupby('Category')[col].mean()

        # Create a bar chart
        bars = plt.bar(means.index, means.values, color=[color_map[cat] for cat in means.index])

        # Customizing plot with titles and labels
        plt.title(f'Mean Distribution of {col} Across Groups')  # Set title to reflect the column name
        plt.xlabel('Category')  # Label for x-axis
        plt.ylabel('Mean Value')  # Label for y-axis

        # Annotate mean values on the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', 
                     ha='center', va='bottom', color='black')

        # Save the plot
        plot_path = os.path.join(output_dir, f'mean_bar_{col}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'Plot saved to {plot_path}')
        
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python synapse_number_Dg.py <export_name> <folder1> <folder2> ... <folder3>")
        print("Each of the folders must include the csv files with the number statistics")
        sys.exit()

    folders = sys.argv[2:]  # Skip the script name, take only file paths
    df = load_data(folders)
    # combined_df = concatenate_dataframes(dataframes)
    
    if not os.path.exists(sys.argv[1]):
        os.makedirs(sys.argv[1])
    
    df.to_csv( os.path.join(sys.argv[1], sys.argv[1] + ".csv") )
    
    print("Creating histograms...")
    grouped_df = prepare_data(df)
    grouped_df.to_csv( os.path.join(sys.argv[1], sys.argv[1] + "grouped_avg" + ".csv"))
    # plot_interactive_histograms(df,sys.argv[1])
    plot_box(grouped_df,sys.argv[1])
    # plot_histograms(df,sys.argv[1])
    print("All data merged and saved to " + sys.argv[1] +".")
