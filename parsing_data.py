import pandas as pd
import datetime as DT
import numpy as np
import pyproj
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

# Parameters
DATA = '../parking_citation_2020_2022.csv'
OUTPUT_CLUSTER_IMG = 'cluster.png'
CLEANED_CSV = '../cleaned_2020_2022_parking_citation.csv'
TOP10_VIOLATIONS_CSV = '../top10_violations_2020_2022.csv'

# Param for Kmeas
NUM_CLUSTERS = 1500

# Param for Density-Based Spatial Clustering of Applications with Noise, SBSCAN
EPS = 0.005  # The maximum distance between two samples for one to be considered as in the neighborhood of the other
MIN_SAMPLES = 100  # The number of samples in a neighborhood for a point to be considered as a core point

# Set up logging
log_filename = f"parsing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_filename)

def visualize_clusters(filtered_data, fig_name):
    logging.info("Visualizing clusters...")
    total_clusters = filtered_data['Cluster'].nunique()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Longitude', y='Latitude', hue='Cluster', data=filtered_data, palette='viridis', s=1, alpha=0.5)
    clusters_text = f"Total number of clusters: {total_clusters}"
    plt.text(0.05, 0.90, clusters_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.title('Visualization of Clusters based on Latitude and Longitude (Longitude < 44)')
    plt.savefig(fig_name)
    logging.info(f"Cluster visualization saved as {fig_name}")

def run():
    logging.info("Starting script...")
    
    df = pd.read_csv(DATA)
    
    # updating formatting so that I can translate issue date to datetime
    df['Issue Date'] = df[df['Issue Date'].notnull()]['Issue Date'].apply(lambda x: x.split('T')[0])
    df['Issue Date'] = pd.to_datetime(df['Issue Date'], infer_datetime_format=True)
    df['Day of Week'] = df['Issue Date'].dt.day_name()

    # pad anything that is less than 4 digits then isolate just the hours
    df['Issue time'] = df['Issue time'].astype(str)
    df['Issue time'] = df['Issue time'].apply(lambda x: x.split('.')[0])
    df['Issue time'] = df[df['Issue time'].notnull()]['Issue time'].apply(lambda x: x.zfill(4))
    df['Issue Hour'] = df[df['Issue time']!='0nan']['Issue time'].apply(lambda x: DT.datetime.strptime(x, '%H%M').hour)
    

    # clean lat lon data
    df['Latitude'] = np.where(df['Latitude'] == 99999.000, np.nan, df['Latitude'])
    df['Longitude'] = np.where(df['Longitude'] == 99999.000, np.nan, df['Longitude'])

    # string for ticket number
    df['Ticket number'] = df['Ticket number'].astype(str)

    # Updating the Lat Lon from regional Lambert Conformal Conic projection to a global WGS 84 (latitude-longitude) 
    pm = '+proj=lcc +lat_1=34.03333333333333 +lat_2=35.46666666666667 +lat_0=33.5 +lon_0=-118 +x_0=2000000 +y_0=500000.0000000002 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs'
    x1m, y1m = df['Latitude'].values, df['Longitude'].values
    x2m, y2m = pyproj.transform(pyproj.Proj(pm, preserve_units=True), pyproj.Proj("+init=epsg:4326"), x1m, y1m)
    df['Latitude'] = x2m
    df['Longitude'] = y2m

    # With some preliminary result from training, I selected these features to better manage the memory use in the training script
    data_df = df[['RP State Plate', 'Make', 'Body Style Description', 'Color Description', 'Agency Description', 'Issue Hour', 'Latitude', 'Longitude', 'Violation Description']]
    data_df = data_df[~data_df.isna().any(axis=1) & (data_df.Longitude < 44)]

    # KMeans clustering. Maybe try Grid-based clustering later?
    coordinates = data_df[['Latitude', 'Longitude']].values
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0).fit(coordinates)
    data_df['Cluster'] = kmeans.labels_

    # # Grid-based clustering using DBSCAN
    # coordinates = data_df[['Latitude', 'Longitude']].values
    # dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(coordinates)
    # data_df['Cluster'] = dbscan.labels_
    
    logging.info("Clustering done.")
    
    visualize_clusters(data_df, OUTPUT_CLUSTER_IMG)
    
    data_df.to_csv(CLEANED_CSV, index=False)
    logging.info(f"Cleaned data saved as {CLEANED_CSV}")
    
    top10_violations = data_df['Violation Description'].value_counts().index[:10]
    top10_df = data_df[data_df['Violation Description'].isin(top10_violations)]
    top10_df.to_csv(TOP10_VIOLATIONS_CSV, index=False)
    logging.info(f"Top 10 violations data saved as {TOP10_VIOLATIONS_CSV}")

if __name__ == "__main__":
    run()
