from data_preprocessing import df,y_train,X_train,wcss
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Count plot of dataset
plt.figure(figsize=(10,6))
sns.countplot(x="diabetes",data=df)
plt.title("Class distribution")
plt.xlabel("Diabetes")
plt.ylabel("Count")
plt.show()

#Correlation matrix 
corr_matrix=df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
plt.title("Correlation matrix ")
plt.show()

#Count plot after resampling
# plt.figure(figsize=(10,8))
# sns.countplot(x=y_train)
# plt.title("Resampled class distribution")
# plt.xlabel("Diabetes class")
# plt.ylabel("Count")   
# plt.show()



numerical_features=['age','bmi','HbA1c_level','blood_glucose_level']
fig,axes=plt.subplots(2,2,figsize=(12,10))
fig.suptitle("Histogram of Numerical features",fontsize=16)

for i, feature in enumerate(numerical_features):
    ax = axes[i // 2, i % 2]
    
    sns.histplot(X_train[feature], ax=ax, kde=True, bins=20)
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(f"{feature}")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df)  # df is your entire DataFrame
plt.title('Boxplot of Features in Full Dataset')

plt.show()

# plt.figure(figsize=(10,8))
# sns.pairplot(df,hue='diabetes')
# plt.show()
