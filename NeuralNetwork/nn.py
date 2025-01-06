import pandas as pd
from imblearn.over_sampling import KMeansSMOTE
from keras.api._tf_keras.keras.layers import Dense
from keras.api._tf_keras.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pymongo import MongoClient
import os 
from dotenv import load_dotenv

load_dotenv()

uri=os.getenv("MONGO_DB_URI")
if uri is None:
    print("MONGODB_URI not found")
    exit()
client = MongoClient(uri)

try:
    client.admin.command('ping')  # Test connection
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
    exit()
    
database = client["diabetes_db"]
collection = database["diabetes_data"]


cursor = collection.find()
df = pd.DataFrame(list(cursor))
df.drop(columns=['_id'],inplace=True)
client.close()

df.drop_duplicates(inplace=True)
scaler=StandardScaler()
# df = pd.get_dummies(df, columns=['smoking_history'], drop_first=True)
X=df.drop(columns=['diabetes','hypertension','heart_disease','gender','smoking_history'])
y=df['diabetes']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
smote=KMeansSMOTE(sampling_strategy='auto',random_state=42)
X_train_res,y_train_res=smote.fit_resample(X_train,y_train)
print(f"Original class distribution in y_train: {y_train.value_counts()}")
print(f"Resampled class distribution in y_train: {pd.Series(y_train_res).value_counts()}")
X_train_scaled=scaler.fit_transform(X_train_res)
X_test_scaled=scaler.transform(X_test)

model=Sequential([
    Dense(64,input_dim=X_train_scaled.shape[1]),
    Dense(32,activation='relu'),
    Dense(16,activation='relu'),
    Dense(1,activation='sigmoid'),
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train_scaled,y_train_res,epochs=10,batch_size=32,validation_data=(X_test_scaled,y_test))

y_pred_prob=model.predict(X_test_scaled)
y_pred=(y_pred_prob>=0.25).astype(int)

print(classification_report(y_test,y_pred))