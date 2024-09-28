import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Correct file path for the dataset
file_path = '/Users/peterzhang/Desktop/Dataset (ATS)-1.csv'

# Print the file path for debugging
print(f"Looking for file at: {file_path}")

# Check if file exists
if os.path.exists(file_path):
    print("File exists.")
    
    # Load the dataset
    dataset = pd.read_csv(file_path)
    
    # Display the first few rows of the dataset to verify loading
    print("Initial Dataset:")
    print(dataset.head())
    
    # Check for missing values
    missing_data = dataset.isnull().sum()
    print("\nMissing Data Summary:")
    print(missing_data)
    
    # Convert 'Churn' column to binary
    dataset['Churn'] = dataset['Churn'].map({'Yes': 1, 'No': 0})
    
    # Perform one-hot encoding for categorical variables
    categorical_columns = ['gender', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract']
    dataset_encoded = pd.get_dummies(dataset, columns=categorical_columns, drop_first=True)
    
    # Feature scaling and normalization for numerical columns
    scaler = StandardScaler()
    dataset_encoded[['MonthlyCharges', 'tenure']] = scaler.fit_transform(dataset_encoded[['MonthlyCharges', 'tenure']])
    
    # Display the processed dataset
    print("\nProcessed Dataset:")
    print(dataset_encoded.head())
    
else:
    print("File not found. Please check the file path.")
    import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# File path for the original dataset
file_path = '/Users/peterzhang/Desktop/Dataset (ATS)-1.csv'

# Check if file exists
if os.path.exists(file_path):
    print("File exists.")
    
    # Load the dataset
    dataset = pd.read_csv(file_path)
    
    # Preprocessing steps (as per previous script)
    dataset['Churn'] = dataset['Churn'].map({'Yes': 1, 'No': 0})
    categorical_columns = ['gender', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract']
    dataset_encoded = pd.get_dummies(dataset, columns=categorical_columns, drop_first=True)
    scaler = StandardScaler()
    dataset_encoded[['MonthlyCharges', 'tenure']] = scaler.fit_transform(dataset_encoded[['MonthlyCharges', 'tenure']])
    
    # Save the preprocessed dataset to the "Data_Preparation" folder
    dataset_encoded.to_csv('/Users/peterzhang/Desktop/Data_Preparation/preprocessed_dataset.csv', index=False)
    print("Preprocessed dataset saved.")
    
else:
    print("File not found. Please check the file path.")

from sklearn.model_selection import train_test_split

# Splitting the dataset into training and testing sets
X = dataset_encoded.drop('Churn', axis=1)  # Features
y = dataset_encoded['Churn']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save training and testing sets to separate CSV files
X_train.to_csv('/Users/peterzhang/Desktop/Data_Preparation/X_train.csv', index=False)
X_test.to_csv('/Users/peterzhang/Desktop/Data_Preparation/X_test.csv', index=False)
y_train.to_csv('/Users/peterzhang/Desktop/Data_Preparation/y_train.csv', index=False)
y_test.to_csv('/Users/peterzhang/Desktop/Data_Preparation/y_test.csv', index=False)

print("Training and testing sets saved.")
