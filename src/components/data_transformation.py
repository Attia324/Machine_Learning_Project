import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_obj(self):
        try:
            # define numerical and categorical features
            numeric_features = [
                'INCOME', 'SAVINGS', 'DEBT', 'R_SAVINGS_INCOME', 'R_DEBT_INCOME',
                'R_DEBT_SAVINGS', 'T_CLOTHING_12', 'T_CLOTHING_6', 'R_CLOTHING',
                'R_CLOTHING_INCOME', 'R_CLOTHING_SAVINGS', 'R_CLOTHING_DEBT',
                'T_EDUCATION_12', 'T_EDUCATION_6', 'R_EDUCATION',
                'R_EDUCATION_INCOME', 'R_EDUCATION_SAVINGS', 'R_EDUCATION_DEBT',
                'T_ENTERTAINMENT_12', 'T_ENTERTAINMENT_6', 'R_ENTERTAINMENT',
                'R_ENTERTAINMENT_INCOME', 'R_ENTERTAINMENT_SAVINGS',
                'R_ENTERTAINMENT_DEBT', 'T_FINES_12', 'T_FINES_6', 'R_FINES',
                'R_FINES_INCOME', 'R_FINES_SAVINGS', 'R_FINES_DEBT',
                'T_GAMBLING_12', 'T_GAMBLING_6', 'R_GAMBLING',
                'R_GAMBLING_INCOME', 'R_GAMBLING_SAVINGS', 'R_GAMBLING_DEBT',
                'T_GROCERIES_12', 'T_GROCERIES_6', 'R_GROCERIES',
                'R_GROCERIES_INCOME', 'R_GROCERIES_SAVINGS', 'R_GROCERIES_DEBT',
                'T_HEALTH_12', 'T_HEALTH_6', 'R_HEALTH', 'R_HEALTH_INCOME',
                'R_HEALTH_SAVINGS', 'R_HEALTH_DEBT', 'T_HOUSING_12', 'T_HOUSING_6',
                'R_HOUSING', 'R_HOUSING_INCOME', 'R_HOUSING_SAVINGS', 'R_HOUSING_DEBT',
                'T_TAX_12', 'T_TAX_6', 'R_TAX', 'R_TAX_INCOME', 'R_TAX_SAVINGS',
                'R_TAX_DEBT', 'T_TRAVEL_12', 'T_TRAVEL_6', 'R_TRAVEL',
                'R_TRAVEL_INCOME', 'R_TRAVEL_SAVINGS', 'R_TRAVEL_DEBT',
                'T_UTILITIES_12', 'T_UTILITIES_6', 'R_UTILITIES',
                'R_UTILITIES_INCOME', 'R_UTILITIES_SAVINGS', 'R_UTILITIES_DEBT',
                'T_EXPENDITURE_12', 'T_EXPENDITURE_6', 'R_EXPENDITURE',
                'R_EXPENDITURE_INCOME', 'R_EXPENDITURE_SAVINGS',
                'R_EXPENDITURE_DEBT'
            ]
            categorical_features = ['CAT_GAMBLING']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical and categorical pipelines created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numeric_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completely")

            target_column_name = "CREDIT_SCORE"

            preprocessing_obj = self.get_transformer_obj()

            input_feature_train_df = train_df.drop(columns=[target_column_name, "CUST_ID"], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name, "CUST_ID"], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saving preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
