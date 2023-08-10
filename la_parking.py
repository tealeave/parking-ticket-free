import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import precision_score, recall_score
# from sklearn.utils import class_weight
import time

start_time = time.time()

# Check cuda
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


# 1. Data Preparation
# ---------------------
data = pd.read_csv("/dfs6/pub/ddlin/projects/parking_citation/top10_violations_2020_2022.csv")
features = [ 'date of week','Issue Hour', 'Cluster']
X = data[features]

# One-hot encode all features
ohe = OneHotEncoder(sparse=False, dtype=int)
X_encoded = ohe.fit_transform(X)
encoded_columns = ohe.get_feature_names_out(features)  # Get column names for the encoded columns
X = pd.DataFrame(X_encoded, columns=encoded_columns)

# Extract target variable and encode it
y = LabelEncoder().fit_transform(data['Violation Description'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Now, after the split, calculate the class weights
# class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 2. Model Definition
# ---------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_dim)  # Modified to directly connect to output

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)  # Directly output without additional activation
        return x


model = SimpleNN(X_train.shape[1], len(set(y_train))).to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initializing lists to store precision and recall values for each epoch
epoch_precisions = []
epoch_recalls = []
epochs = 10
# 3. Training
# ------------
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # 4. Evaluation
    # ---------------
    with torch.no_grad():
        y_pred = model(X_train_tensor)
        _, predicted = torch.max(y_pred, 1)
        
        # Convert tensors to numpy arrays for sklearn metrics
        true_labels = y_train_tensor.cpu().numpy()
        pred_labels = predicted.cpu().numpy()
        
        # Calculate precision and recall
        precision = precision_score(true_labels, pred_labels, average='macro')
        recall = recall_score(true_labels, pred_labels, average='macro')
        epoch_precisions.append(precision)
        epoch_recalls.append(recall)

        accuracy = (predicted == y_train_tensor).sum().item() / len(y_train_tensor)
        print(f"Training Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # prevent memory issue
    torch.cuda.empty_cache()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")