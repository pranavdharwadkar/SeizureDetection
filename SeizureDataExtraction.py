import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the dataset
file_path = '/Users/pranav/Downloads/Depression_VS_Healthy_Detection_EEG - data.csv'
data = pd.read_csv(file_path)

# Filter the necessary columns (Condition + columns ending with FP1, FP2, T3, T4 for delta, beta, alpha, theta)
columns_of_interest = ['Condition']
bands = ['delta', 'beta', 'alpha', 'theta']
electrodes = ['FP1', 'FP2', 'T3', 'T4']

for band in bands:
    for electrode in electrodes:
        # Add columns that match the pattern of interest
        columns_of_interest.extend([col for col in data.columns if f'{band}' in col and col.endswith(electrode)])

# Filter the dataset to include only these columns
filtered_data = data[columns_of_interest]

# Convert all values in the filtered data (except 'Condition') to numeric, forcing errors to NaN
filtered_data_numeric = filtered_data.copy()

# Convert all the band/electrode columns to numeric, ignore errors
for col in filtered_data_numeric.columns[1:]:
    filtered_data_numeric[col] = pd.to_numeric(filtered_data_numeric[col], errors='coerce')

# Function to compute the mean across FP1, FP2, T3, T4 for each band
def compute_band_means(row):
    means = []
    for band in bands:
        # Extract the relevant electrode values for this band
        electrode_values = [row[col] for col in row.index if band in col]
        means.append(np.mean(electrode_values))
    return means

# Apply the function to each row to calculate mean values for delta, beta, alpha, theta
features = filtered_data_numeric.apply(lambda row: compute_band_means(row), axis=1)

# Convert features to a numpy array
features = np.vstack(features.values)

# Extract the labels (conditions: Depression or Healthy)
labels = filtered_data_numeric['Condition'].map(lambda x: 1 if x == 'Depression' else 0).values

# Display first 5 rows of features and labels
print(features[:5], labels[:5])

features_cleaned = np.where(np.isnan(features), np.nanmean(features, axis=0), features)


X_train, X_test, y_train, y_test = train_test_split(features_cleaned, labels, test_size=0.2, random_state=42)

# Step 2: Standardize the features (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train a logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 4: Make predictions on the test data
y_pred = model.predict(X_test_scaled)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


# Step 4: Define the models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Naive Bayes': GaussianNB()
}

# Step 5: Train and evaluate each model
results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'class_report': class_report
    }

# Step 6: Compare results
for name, result in results.items():
    print(f"Model: {name}")
    print(f"Accuracy: {result['accuracy']}")
    print("Confusion Matrix:")
    print(result['conf_matrix'])
    print("Classification Report:")
    print(result['class_report'])
    print("="*50)
