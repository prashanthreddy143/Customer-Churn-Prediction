import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

class DataLoader:
    """
    A class to load and preprocess customer churn data
    """
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(self.filepath)
            print(f"Data loaded successfully! Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"Error: File {self.filepath} not found!")
            return None
    
    def explore_data(self):
        """Display basic information about the dataset"""
        if self.data is None:
            print("Please load data first using load_data()")
            return
        
        print("\n=== Dataset Info ===")
        print(f"Shape: {self.data.shape}")
        print(f"\nFirst few rows:\n{self.data.head()}")
        print(f"\nData types:\n{self.data.dtypes}")
        print(f"\nMissing values:\n{self.data.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.data.describe()}")
    
    def handle_missing_values(self, strategy='drop'):
        """Handle missing values in the dataset"""
        if strategy == 'drop':
            self.data = self.data.dropna()
        elif strategy == 'mean':
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        print(f"Missing values handled. New shape: {self.data.shape}")
    
    def encode_categorical_features(self):
        """Encode categorical features using LabelEncoder"""
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'Churn':  # Don't encode target variable here
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"Categorical features encoded. Total encoders: {len(self.label_encoders)}")
    
    def encode_target_variable(self, target_col='Churn'):
        """Encode the target variable"""
        if target_col in self.data.columns:
            le = LabelEncoder()
            self.data[target_col] = le.fit_transform(self.data[target_col].astype(str))
            self.label_encoders[target_col] = le
            print(f"Target variable '{target_col}' encoded.")
    
    def split_features_target(self, target_col='Churn'):
        """Split data into features (X) and target (y)"""
        if target_col not in self.data.columns:
            print(f"Error: Target column '{target_col}' not found!")
            return None, None
        
        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def get_processed_data(self, target_col='Churn', test_size=0.2):
        """Complete preprocessing pipeline"""
        from sklearn.model_selection import train_test_split
        
        # Handle missing values
        self.handle_missing_values()
        
        # Encode categorical features
        self.encode_categorical_features()
        
        # Encode target variable
        self.encode_target_variable(target_col)
        
        # Split features and target
        X, y = self.split_features_target(target_col)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
