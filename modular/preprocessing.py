import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from scipy.stats import chi2_contingency
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

    # columns_to_drop = ['gender', 'hypertension', 'heart_disease', 'smoking_history']  # Replace with actual feature names
    # df.drop(columns=columns_to_drop, axis=1, inplace=True)
    # scaler=StandardScaler()
    # df_scaled=scaler.fit_transform(df.drop(columns=['diabetes']))
    # pca=PCA()
    # pca.fit(df_scaled)
    # print("Explained Variance Ratio (Cumulative):")
    # print(np.cumsum(pca.explained_variance_ratio_))

    # X=pd.DataFrame(df_scaled)
    # y=df['diabetes']
    X = df.drop(columns=['diabetes'], axis=1)
    y = df['diabetes']

    return X, y

def split_data(X, y, test_size=0.2, random_state=22):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# def perform_chi_square_test(df, target_col):
#     categorical_columns = df.select_dtypes(include=['object', 'int']).columns
#     results = {}

#     for col in categorical_columns:
#         if col == target_col:
#             continue
        
#         # Create a contingency table
#         contingency_table = pd.crosstab(df[col], df[target_col])
        
#         # Perform the Chi-Square test
#         chi2, p, dof, expected = chi2_contingency(contingency_table)
        
#         results[col] = {
#             'Chi-Square Statistic': chi2,
#             'p-value': p,
#             'Degrees of Freedom': dof,
#             'Dependent': p < 0.05  # True if dependent on the target
#         }

#     print(results)