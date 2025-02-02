import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import boxcox
from sklearn.metrics import roc_curve,roc_auc_score
from Config.configuration import log_info,log_error



#Count plot for target variable
def plot_class_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x="diabetes", data=df)
    plt.title("Class distribution")
    plt.xlabel("Diabetes")
    plt.ylabel("Count")
    plt.show()
    log_info("Count plot is visualized")

#Correlation matrix to check correlation and multicollinearity between all features including target
def plot_correlation_matrix(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation matrix")
    plt.show()
    log_info("Correlation matrix is visualized")

#Histplot for checking distribution of continuous features.
def plot_histograms(X_train, numerical_features):
    X_train_transformed = X_train.copy()  
    # Apply log transformation to 'HbA1c_level' and 'blood_glucose_level', and Box-Cox to 'bmi'
    X_train_transformed['HbA1c_level'] = np.log1p(X_train['HbA1c_level'])
    X_train_transformed['blood_glucose_level'] = np.log1p(X_train['blood_glucose_level'])
    X_train_transformed['bmi'], _ = boxcox(X_train['bmi'] + 1)  # Add 1 to avoid zero

    fig, axes = plt.subplots(2, 4, figsize=(16, 12))
    fig.suptitle("Histograms of Numerical Features Before and After Transformation", fontsize=16)

    for i, feature in enumerate(numerical_features):
        # Before transformation
        ax = axes[0, i]
        sns.histplot(X_train[feature], ax=ax, kde=True, bins=20)
        ax.set_title(f"Original {feature}")
        ax.set_xlabel(f"{feature}")
        ax.set_ylabel("Frequency")

        # After transformation
        ax = axes[1, i]
        sns.histplot(X_train_transformed[feature], ax=ax, kde=True, bins=20)
        ax.set_title(f"Transformed {feature}")
        ax.set_xlabel(f"{feature}")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    log_info("Histplots are visualized")

#Boxplot to check for outliers and ranges of features
def plot_boxplot(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df)
    plt.title('Boxplot of Features in Full Dataset')
    plt.show()
    log_info("Box plot is visualized")

#Pairplot for visualizing relationships between multiple variables
def plot_pairplot(df,target_column='diabetes'):
    sns.pairplot(df,hue=target_column,markers=["o","s"],palette="coolwarm")
    plt.suptitle("Pairplot of features colored by diabetes status",y=1.02,fontsize=16)
    plt.show()
    log_info("Pairplot is visualized")

#Roc curve to evaluate performance of binary classification by plotting TPR against FPR  
def calculate_auc_roc(y_true, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba[:, 1])  # For binary classification, use the probabilities of class 1
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line (no discrimination)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    return auc
