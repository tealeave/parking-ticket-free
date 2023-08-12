import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_clusters():
    # Load Data
    data = pd.read_csv('/dfs6/pub/ddlin/projects/parking_citation/top10_violations_2020_2022.csv')

    # Set up the plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=data, x='Longitude', y='Latitude', hue='Cluster', palette='viridis', s=60, alpha=0.5)
    plt.title('Latitude vs Longitude Clusters')

    # Save the plot as 'cluster.png'
    plt.savefig('cluster.png')
    plt.close()

if __name__ == "__main__":
    visualize_clusters()
