import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

df=pd.read_csv("data/diabetes_prediction_dataset.csv")
# print(df.head())

# print(df.shape) (100000,9)

#Checking for null values
# print(df.isnull().sum())  #No null values found in all columns

#Checking for duplicates
# print(df.duplicated().sum()) Found 3854 duplicates
df.drop_duplicates(inplace=True)
# print("Shape after dropping duplicates: ",df.shape) #(96146,9)

# print(df.info()) 2 features: gender,smoking_history are categorical, rest are numerical

# print(df.nunique())
df['age']=df['age'].apply(lambda x:1 if x<1 else x)
# print(df.describe())

df['smoking_history']=df['smoking_history'].replace({
    'never':'never_or_ever',
    'ever':'never_or_ever',
    'former':'former_or_not_current',
    'not current':'former_or_not_current'
})

# print(df['smoking_history'].value_counts())




categorical=['gender','smoking_history']

le=LabelEncoder()
for feature in categorical:
    df[feature]=le.fit_transform(df[feature])

X=df.drop(columns=['diabetes'],axis=1)
y=df['diabetes']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=22)


scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)

# wcss=[]
# for i in range(2,11):
#     kmeans=KMeans(n_clusters=i,random_state=22,init='k-means++',max_iter=300,n_init=10)
#     kmeans.fit(X_train_scaled)
#     wcss.append(kmeans.inertia_)

# plt.figure(figsize=(10,8))
# plt.plot(range(2,11),wcss)
# plt.show()

# silhoutte_score=[]
# for i in range(2,11):
#     k_means=KMeans(n_clusters=i,random_state=22)
#     k_means.fit(X_train_scaled)
#     silhoutte_score.append(silhouette_score(X_train_scaled,k_means.labels_))

# plt.figure(figsize=(10,7))
# plt.plot(range(2,11),silhoutte_score)
# plt.show()


# le=LabelEncoder()
# for feature in categorical:
#     X_train[feature]=le.fit_transform(X_train[feature])
#     X_test[feature]=le.transform(X_test[feature])

# print(X_train.head())
# print(X_test.head())


# print(X_train_scaled.head)
# print(X_test_scaled.head)

#Balancing class variable
# undersampler=RandomUnderSampler(sampling_strategy='auto',random_state=22)

# X_train_resampled,y_train_resampled=undersampler.fit_resample(X_train,y_train)

# oversampler=SMOTE(sampling_strategy='auto',random_state=22)
# X_train_resampled,y_train_resampled=oversampler.fit_resample(X_train,y_train)

# print(f"Original class distribution in y_train: {y_train.value_counts()}")
# print(f"Resampled class distribution in y_train: {y_train_resampled.value_counts()}")

# scaler=StandardScaler()
# X_train_scaled=pd.DataFrame(scaler.fit_transform(X_train_resampled),columns=X_train.columns)
# # X_train_scaled=pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
# X_test_scaled=pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)

# print(X_test_scaled.head(15))
# print(X_train_resampled.describe())
