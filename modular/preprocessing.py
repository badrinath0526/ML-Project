import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox,chi2_contingency
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
# def apply_transformations(X):
    
#     X = X.copy()  # Avoid modifying the original dataset

#     # Apply log transformation
#     X['HbA1c_level'] = np.log1p(X['HbA1c_level'])
#     X['blood_glucose_level'] = np.log1p(X['blood_glucose_level'])

#     # Apply Box-Cox transformation (ensure non-negative data)
#     X['bmi'], _ = boxcox(X['bmi'] + 1)  # Add 1 to avoid zero
    
#     return X

def preprocess_features(df):
    df['age'] = df['age'].apply(lambda x: 1 if x < 1 else x)

    columns_to_drop = ['gender', 'hypertension', 'heart_disease', 'smoking_history']  # Replace with actual feature names
    df.drop(columns=columns_to_drop, axis=1, inplace=True)


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

    # X=apply_transformations(X)

    return X, y

def split_data(X, y, test_size=0.2, random_state=22):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # # Apply Box-Cox transformation using parameters fitted on training data
    # X_train = apply_transformations(X_train)
    # X_test = apply_transformations(X_test)

    # return X_train, X_test, y_train, y_test


# def perform_chi_square_test(df, target_col):
#     # Only select categorical columns, 'smoking_history' and 'gender' are categorical
#     categorical_columns = ['gender', 'smoking_history']
#     results = {}

#     for col in categorical_columns:
#         if col == target_col:
#             continue
        
#         # Create a contingency table for Chi-Square test
#         contingency_table = pd.crosstab(df[col], df[target_col])
        
#         # Perform the Chi-Square test
#         chi2, p, dof, expected = chi2_contingency(contingency_table)
        
#         results[col] = {
#             'Chi-Square Statistic': chi2,
#             'p-value': p,
#             'Degrees of Freedom': dof,
#             'Dependent': p < 0.05  # True if dependent on the target
#         }

#     print("Chi-Square Test Results for Categorical Variables:")
#     print(results)

# def perform_anova_f_test(X, y):
#     # Apply ANOVA F-test to each feature
#     f_values, p_values = f_classif(X, y)
    
#     anova_results = {}
#     for i, feature in enumerate(X.columns):
#         anova_results[feature] = {
#             'F-Statistic': f_values[i],
#             'p-value': p_values[i],
#             'Significant': p_values[i] < 0.05
#         }
    
#     print("ANOVA F-test Results:")
#     print(anova_results)
