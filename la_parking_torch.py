"""
Script for training a neural network model on parking citation data.

Data Source: 
The dataset used in this script is obtained from the Los Angeles city's public database.
URL: https://data.lacity.org/Transportation/Parking-Citations/wjz9-h9np

The script utilizes PyTorch for model definition, training, and evaluation, and 
Scikit-learn for data preprocessing and metrics calculation.
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import precision_score, recall_score
import time
import os

# Parameters
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
DATA_PATH = "/dfs6/pub/ddlin/projects/parking_citation/top10_violations_2020_2022.csv"
# Set memory management parameters for PyTorch CUDA
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=256,min_split_size_mb=4,not_reuse_first=1'



class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def run():
    start_time = time.time()

    # Check CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")

    # Data Preparation
    data = pd.read_csv(DATA_PATH)
    features = ['date of week', 'Issue Hour', 'Cluster']
    X = data[features]
    ohe = OneHotEncoder(sparse=False, dtype=int)
    X_encoded = ohe.fit_transform(X)
    encoded_columns = ohe.get_feature_names_out(features)
    X = pd.DataFrame(X_encoded, columns=encoded_columns)
    y = LabelEncoder().fit_transform(data['Violation Description'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model Definition & Training
    model = SimpleNN(X_train.shape[1], len(set(y_train))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epoch_precisions, epoch_recalls = [], []
    for epoch in range(EPOCHS):
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")

        # Evaluation
        with torch.no_grad():
            y_pred = model(X_train_tensor)
            _, predicted = torch.max(y_pred, 1)
            true_labels = y_train_tensor.cpu().numpy()
            pred_labels = predicted.cpu().numpy()
            precision = precision_score(true_labels, pred_labels, average='macro')
            recall = recall_score(true_labels, pred_labels, average='macro')
            epoch_precisions.append(precision)
            epoch_recalls.append(recall)
            accuracy = (predicted == y_train_tensor).sum().item() / len(y_train_tensor)
            print(f"Training Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        torch.cuda.empty_cache()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    run()
