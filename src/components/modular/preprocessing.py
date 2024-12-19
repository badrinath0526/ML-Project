import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
import joblib

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    # Dropping duplicates
    df.drop_duplicates(inplace=True)
    

    # Handling categorical features
    df['smoking_history'] = df['smoking_history'].replace({
        'never': 'never',
        'ever': 'never',
        'former': 'past',
        'not current': 'past'
    })

    # Label encoding for categorical columns
    le_gender = LabelEncoder()
    le_smoking_history=LabelEncoder()
    
    df['gender']=le_gender.fit_transform(df['gender'])
    df['smoking_history']=le_smoking_history.fit_transform(df['smoking_history'])

    joblib.dump(le_gender,'le_gender.pkl')
    joblib.dump(le_smoking_history,'le_smoking_history.pkl')


    return df   

def preprocess_features(df):
    df['age'] = df['age'].apply(lambda x: 1 if x < 1 else x)
    X = df.drop(columns=['diabetes'], axis=1)
    y = df['diabetes']

    return X, y

def split_data(X, y, test_size=0.2, random_state=22):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
