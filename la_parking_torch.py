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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time
import logging
import datetime

# logging
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(filename=f"training_{current_time}.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
BATCH_SIZE = 1024
EPOCHS = 5
LEARNING_RATE = 0.002
DATA_PATH = "/dfs6/pub/ddlin/projects/parking_citation/top10_violations_2020_2022_km3000.csv"


# The top 10 citations, these are unbalanced classes
counts = {
    "NO PARK/STREET CLEAN": 1192320,
    "METER EXP.": 830813,
    "RED ZONE": 524986,
    "PREFERENTIAL PARKING": 360569,
    "DISPLAY OF TABS": 276076,
    "NO PARKING": 191136,
    "DISPLAY OF PLATES": 153174,
    "PARKED OVER TIME LIMIT": 111974,
    "NO STOP/STANDING": 109257,
    "STANDNG IN ALLEY": 89263
}

total_samples = sum(counts.values())
num_classes = len(counts)
class_weights = {cls: total_samples / (num_classes * count) for cls, count in counts.items()}

# Convert class_weights dictionary to a tensor in the correct order for the dataset
weights_tensor = torch.tensor([class_weights[cls] for cls in sorted(counts.keys())])

class SimpleNN(nn.Module):
    """A feedforward neural network with GELU activation and Batch Normalization."""
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.layer3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.layer4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.layer5 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = F.relu(self.bn3(self.layer3(x)))
        x = F.relu(self.bn4(self.layer4(x)))
        x = self.layer5(x)
        return x

def load_data():
    """Loads and preprocesses the dataset."""
    data = pd.read_csv(DATA_PATH)
    print((f'Mem after reading csv: {data.info(memory_usage="deep")}'))
    
    features = ['Agency Description', 'Day of Week', 'Issue Hour', 'Cluster']
    X = data[features]
    ohe = OneHotEncoder(sparse=False, dtype=int)
    X_encoded = ohe.fit_transform(X)
    logging.info('Done with X one hot encoding')
    
    encoded_columns = ohe.get_feature_names_out(features)
    X = pd.DataFrame(X_encoded, columns=encoded_columns).apply(pd.to_numeric, downcast='integer')
    logging.info('Done with X processing')
    
    y = LabelEncoder().fit_transform(data['Violation Description'])

    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate(model, dataloader, device, loss_fn):
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = loss_fn(outputs, target)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    average_loss = total_loss / len(dataloader)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    return accuracy, average_loss

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
    logging.info('Done with train test split')
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Convert test data to tensors
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    # Create data loader for evaluation
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model Definition & Training
    model = SimpleNN(X_train.shape[1], len(set(y_train))).to(device)
    # Log model architecture
    logging.info(f'Model architecture: {model}')  

    # Test if the class weights in the loss function will help 
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        # Training
        for data, target in train_loader:
            logging.debug(f"data dimensions: {data.size()}")
            logging.debug(f"target dimensions: {target.size()}")
            data, target = data.to(device), target.to(device)  # Move batch to GPU
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        logging.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")

        # Evaluation
        train_accuracy, train_loss = evaluate(model, train_loader, device, criterion)
        logging.info(f"Training Accuracy: {train_accuracy:.4f}, Loss: {train_loss:.4f}")

        test_accuracy, test_loss = evaluate(model, test_loader, device, criterion)
        logging.info(f"Test Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")


        # torch.cuda.empty_cache()

    # Serialize the model
    torch.save(model.state_dict(), 'parking_ticket_pred_model.pth')
    logging.info(f"Model saved as parking_ticket_pred_model.pth")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    run()
