import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
import joblib
from Config.configuration import log_info,log_error
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
#Cleans data by droppping duplicates and encoding categorical variables
# def load_and_clean_data(file_path):
#     df = pd.read_csv(file_path)
# 
def load_and_clean_data():
    # MongoDB connection URI
    uri = os.getenv('MONGO_DB_URI')
    if uri is None:
        print("MONGODB_URI not found")
    client = MongoClient(uri)

    try:
        client.admin.command('ping')  # Test connection
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    
    database = client["diabetes_db"]
    collection = database["diabetes_data"]


    # Retrieve the data from MongoDB
    cursor = collection.find()
    df = pd.DataFrame(list(cursor))
    df.drop(columns=['_id'],inplace=True)
    client.close()

    return df

def preprocess_data(df):
    # Dropping duplicates
    df.drop_duplicates(inplace=True)
    log_info("Dropped duplicates")

    # Handling categorical features
    df['smoking_history'] = df['smoking_history'].replace({
        'never': 'never',
        'ever': 'never',
        'former': 'past',
        'not current': 'past'
    })

    # Label encoding for categorical columns
    le_gender = LabelEncoder()
    le_smoking_history = LabelEncoder()

    df['gender'] = le_gender.fit_transform(df['gender'])
    df['smoking_history'] = le_smoking_history.fit_transform(df['smoking_history'])
    log_info("Encoded categorical variables")

    # Save label encoders
    joblib.dump(le_gender, 'le_gender.pkl')
    joblib.dump(le_smoking_history, 'le_smoking_history.pkl')

    return df

# Cleans data by dropping duplicates and encoding categorical variables
def preprocess_data(df):
    # Dropping duplicates
    df.drop_duplicates(inplace=True)
    log_info("Dropped duplicates")

    # Handling categorical features
    df['smoking_history'] = df['smoking_history'].replace({
        'never': 'never',
        'ever': 'never',
        'former': 'past',
        'not current': 'past'
    })

    # Label encoding for categorical columns
    le_gender = LabelEncoder()
    le_smoking_history = LabelEncoder()

    df['gender'] = le_gender.fit_transform(df['gender'])
    df['smoking_history'] = le_smoking_history.fit_transform(df['smoking_history'])
    log_info("Encoded categorical variables")

    # Save label encoders
    joblib.dump(le_gender, 'le_gender.pkl')
    joblib.dump(le_smoking_history, 'le_smoking_history.pkl')

    return df


#Preprocesses features
def preprocess_features(df):
    df['age'] = df['age'].apply(lambda x: 1 if x < 1 else x)

    columns_to_drop = ['gender', 'hypertension', 'heart_disease','smoking_history'] 
    df.drop(columns=columns_to_drop, axis=1, inplace=True)

    X = df.drop(columns=['diabetes'], axis=1)
    y = df['diabetes']
    log_info("Dropped unnecessary features")


    return X, y



#Splits the data into training and testing 
def split_data(X, y, test_size=0.2, random_state=22):
    log_info("Split data into train and test")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

    

#Performs chi-square test on categorical variables
def perform_chi_square_test(df, target_col):
    # Only select categorical columns, 'smoking_history' and 'gender' are categorical
    categorical_columns = ['gender', 'smoking_history']
    results = {}

    for col in categorical_columns:
        if col == target_col:
            continue
        
        # Create a contingency table for Chi-Square test
        contingency_table = pd.crosstab(df[col], df[target_col])
        
        # Perform the Chi-Square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        results[col] = {
            'Chi-Square Statistic': chi2,
            'p-value': p,
            'Degrees of Freedom': dof,
            'Dependent': p < 0.05  
        }

    print("Chi-Square Test Results for Categorical Variables:")
    print(results)




