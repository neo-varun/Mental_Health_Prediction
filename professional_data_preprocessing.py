import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'professional_preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.numerical_features = [
            'Age', 'Work Pressure','Job Satisfaction',
            'Work/Study Hours', 'Financial Stress'
        ]
        self.ordinal_features = [
            'Sleep Duration', 'Dietary Habits',
            'Have you ever had suicidal thoughts ?',
            'Family History of Mental Illness'
        ]
        self.nominal_features = [
            'Gender', 'City', 'Profession', 'Degree'
        ]

    def initiate_data_transformation(self):
        sleep_duration = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
        dietary_habits = ['Unhealthy', 'Moderate', 'Healthy']
        suicide_thoughts = ['Yes', 'No']
        family_history = ['Yes', 'No']

        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        ordinal_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinalencoder', OrdinalEncoder(categories=[
                sleep_duration, dietary_habits, suicide_thoughts, family_history
            ])),
            ('scaler', StandardScaler())
        ])

        nominal_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num_pipeline', num_pipeline, self.numerical_features),
            ('ordinal_pipeline', ordinal_pipeline, self.ordinal_features),
            ('nominal_pipeline', nominal_pipeline, self.nominal_features)
        ])

        return preprocessor

    def initialize_data_transformation(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Clean training data
        train_df = train_df[~train_df['Sleep Duration'].isin([
            '2-3 hours', '1-2 hours', '10-11 hours', '40-45 hours',
            'Moderate', '55-66 hours', '3-4 hours', '4-5 hours'
        ])]
        train_df = train_df[~train_df['Dietary Habits'].isin([
            '3', 'Less than Healthy', 'Mihir', '1'
        ])]

        # Clean testing data
        test_df = test_df[~test_df['Sleep Duration'].isin([
            '2-3 hours', '1-2 hours', '10-11 hours', '40-45 hours',
            'Moderate', '55-66 hours', '3-4 hours', '4-5 hours'
        ])]
        test_df = test_df[~test_df['Dietary Habits'].isin([
            '3', 'Less than Healthy', 'Mihir', '1'
        ])]

        # Ensure only valid categories remain in 'Sleep Duration'
        valid_sleep_duration = ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours']
        train_df['Sleep Duration'] = train_df['Sleep Duration'].where(train_df['Sleep Duration'].isin(valid_sleep_duration), np.nan)
        test_df['Sleep Duration'] = test_df['Sleep Duration'].where(test_df['Sleep Duration'].isin(valid_sleep_duration), np.nan)

        # Ensure only valid categories remain in 'Dietary Habits'
        valid_dietary_habits = ['Unhealthy', 'Moderate', 'Healthy']
        train_df['Dietary Habits'] = train_df['Dietary Habits'].where(train_df['Dietary Habits'].isin(valid_dietary_habits), np.nan)
        test_df['Dietary Habits'] = test_df['Dietary Habits'].where(test_df['Dietary Habits'].isin(valid_dietary_habits), np.nan)

        # Drop unnecessary columns
        columns_to_drop = [
            'id', 'Name', 'Working Professional or Student',
            'Academic Pressure', 'CGPA', 'Study Satisfaction'
        ]
        train_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        test_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        target_column_name = 'Depression'
        drop_columns = [target_column_name]

        input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
        target_feature_train_df = train_df[target_column_name]

        input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
        target_feature_test_df = test_df[target_column_name]

        # Validate input data columns
        expected_columns = self.numerical_features + self.ordinal_features + self.nominal_features
        missing_columns = [col for col in expected_columns if col not in train_df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in input data: {missing_columns}")

        preprocessing_obj = self.initiate_data_transformation()

        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

        train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

        # Save the preprocessing object
        os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
        with open(self.data_transformation_config.preprocessor_obj_file_path, 'wb') as f:
            pickle.dump(preprocessing_obj, f)

        return train_arr, test_arr

    def load_and_preprocess(self, data_path):
        """Load and preprocess data for metrics calculation"""
        df = pd.read_csv(data_path)
        
        # Drop unnecessary columns first
        columns_to_drop = [
            'id', 'Name', 'Working Professional or Student',
            'Academic Pressure', 'CGPA', 'Study Satisfaction'
        ]
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        # Get features and target
        target = 'Depression'
        features = df.drop(columns=[target])
        
        # Load preprocessor
        preprocessor_path = os.path.join('artifacts', 'professional_preprocessor.pkl')
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
            
        # Transform features
        X = preprocessor.transform(features)
        y = df[target].values
        
        return X, y