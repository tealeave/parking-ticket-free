"""
Script for training a neural network model on parking citation data.

Data Source: 
The dataset used in this script is obtained from the Los Angeles city's public database.
URL: https://data.lacity.org/Transportation/Parking-Citations/wjz9-h9np

The script utilizes PyTorch for model definition, training, and evaluation, and 
Scikit-learn for data preprocessing and metrics calculation.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import precision_score, recall_score
import time
import logging

# Parameters
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 0.002
DATA_PATH = "/dfs6/pub/ddlin/projects/parking_citation/top10_violations_2020_2022.csv"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=256,min_split_size_mb=4,not_reuse_first=1'

# # The counts from your awk command
# counts = {
#     "NO PARK/STREET CLEAN": 1192320,
#     "METER EXP.": 830813,
#     "RED ZONE": 524986,
#     "PREFERENTIAL PARKING": 360569,
#     "DISPLAY OF TABS": 276076,
#     "NO PARKING": 191136,
#     "DISPLAY OF PLATES": 153174,
#     "PARKED OVER TIME LIMIT": 111974,
#     "NO STOP/STANDING": 109257,
#     "STANDNG IN ALLEY": 89263
# }

# total_samples = sum(counts.values())
# num_classes = len(counts)
# class_weights = {cls: total_samples / (num_classes * count) for cls, count in counts.items()}

# # Convert class_weights dictionary to a tensor in the correct order for the dataset
# weights_tensor = torch.tensor([class_weights[cls] for cls in sorted(counts.keys())])

class SimpleNN(nn.Module):
    """A simple feedforward neural network."""
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x

def load_data():
    """Loads and preprocesses the dataset."""
    data = pd.read_csv(DATA_PATH)
    features = ['date of week', 'Issue Hour', 'Cluster']
    X = data[features]
    ohe = OneHotEncoder(sparse=False, dtype=int)
    X_encoded = ohe.fit_transform(X)
    encoded_columns = ohe.get_feature_names_out(features)
    X = pd.DataFrame(X_encoded, columns=encoded_columns)
    y = LabelEncoder().fit_transform(data['Violation Description'])
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate(model, dataloader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    return accuracy, precision, recall

def run():
    start_time = time.time()

    # Log hyperparameters
    logging.info(f'BATCH_SIZE: {BATCH_SIZE}')
    logging.info(f'EPOCHS: {EPOCHS}')
    logging.info(f'LEARNING_RATE: {LEARNING_RATE}')

    # Check CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info(f"Using device: {device}")

    X_train, X_test, y_train, y_test = load_data()
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Convert test data to tensors
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    # Create data loader for evaluation
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model Definition & Training
    model = SimpleNN(X_train.shape[1], len(set(y_train))).to(device)
    # Log model architecture
    logging.info(f'Model architecture: {model}')  

    # Use the weights in the loss function
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)  # Move batch to GPU
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        logging.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")

        # Evaluation
        train_accuracy, train_precision, train_recall = evaluate(model, train_loader, device)
        logging.info(f"Training Accuracy: {train_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")

        test_accuracy, test_precision, test_recall = evaluate(model, test_loader, device)
        logging.info(f"Test Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

        # torch.cuda.empty_cache()

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    run()
