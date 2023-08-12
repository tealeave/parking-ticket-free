import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = '/dfs6/pub/ddlin/projects/parking_citation/top10_violations_2020_2022.csv'

def visualize_clusters(DATA_PATH):
    # Load Data
    data = pd.read_csv(DATA_PATH)

    # Filter data where Longitude < 44
    filtered_data = data[data['Longitude'] < 44]

    # Calculate total number of unique clusters
    total_clusters = filtered_data['Cluster'].nunique()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', data=filtered_data, palette='viridis', s=1, alpha=0.5)
    
    # Calculate percentage for Longitude < 44 in the original data
    percentage = (len(filtered_data) / len(data)) * 100
    percentage_text = f"Percentage of data that has Longitude < 44 in the original dataset: {percentage:.2f}%"
    clusters_text = f"Total number of clusters: {total_clusters}"
    
    plt.text(0.05, 0.95, percentage_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.05, 0.90, clusters_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

    plt.title('Visualization of Clusters based on Latitude and Longitude (Longitude < 44)')
    plt.savefig('cluster_long_updated.png')

if __name__ == "__main__":
    visualize_clusters()
