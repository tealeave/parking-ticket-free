import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score
import time

# Parameters
DATA_PATH = "/dfs6/pub/ddlin/projects/parking_citation/top10_violations_2020_2022.csv"

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

def run():
    start_time = time.time()

    X_train, X_test, y_train, y_test = load_data()

    # Model Definition & Grid Search
    clf = xgb.XGBClassifier(objective='multi:softprob')
    
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'n_estimators': [ 100, 200, 400]
    }
    
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Print best parameters
    print(f"Best parameters found: {grid_search.best_params_}")

    # Training Evaluation using best model
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(X_train)
    precision_train = precision_score(y_train, y_pred_train, average='macro')
    recall_train = recall_score(y_train, y_pred_train, average='macro')
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print(f"Training Accuracy with best model: {accuracy_train:.4f}, Precision: {precision_train:.4f}, Recall: {recall_train:.4f}")

    # Test Evaluation using best model
    y_pred_test = best_model.predict(X_test)
    precision_test = precision_score(y_test, y_pred_test, average='macro')
    recall_test = recall_score(y_test, y_pred_test, average='macro')
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy with best model: {accuracy_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    run()
