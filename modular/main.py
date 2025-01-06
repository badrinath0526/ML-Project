from preprocessing import load_and_clean_data, preprocess_features, split_data,perform_chi_square_test,preprocess_data
from plotting import plot_class_distribution, plot_correlation_matrix, plot_histograms, plot_boxplot,plot_pairplot
from modeltraining import build_pipeline, grid_search_tuning, evaluate_model,save_model,calculate_training_accuracy,evaluate_roc_auc,perform_rfe
import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from Config.configuration import log_info
import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

df = load_and_clean_data()
df=preprocess_data(df)

# perform_chi_square_test(df,target_col='diabetes')

X, y = preprocess_features(df)

X_train, X_test, y_train, y_test = split_data(X, y)
X_train_reg,X_test_reg,y_train_reg,y_test_reg=split_data(X,y)



# perform_rfe(X_train, y_train)


small_df=df.iloc[30000:60000]

# Visualization

plot_class_distribution(df)
plot_correlation_matrix(df)
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
plot_histograms(X_train, numerical_features)
plot_boxplot(df)
plot_pairplot(small_df,target_column="diabetes")


# Model training and evaluation

pipeline = build_pipeline()
log_info("Data sent into pipeline for training")
grid_search = grid_search_tuning(pipeline, X_train, y_train)
train_accuracy = calculate_training_accuracy(grid_search, X_train, y_train)
print(f"Training Accuracy: {train_accuracy:.4f}")
# get_feature_importance(grid_search)
evaluate_model(grid_search, X_test, y_test)
best_model = grid_search.best_estimator_

evaluate_roc_auc(best_model, X_test, y_test)

save_model(grid_search)

#Regression model for generating continuous values and retraining to get risk percentage

model=LinearRegression()
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train_reg)
X_test_scaled=scaler.transform(X_test_reg)
model.fit(X_train_scaled,y_train_reg)
y_pred_train=model.predict(X_train_scaled)
y_pred_prob_train=1/(1+np.exp(-y_pred_train))
y_pred_test=model.predict(X_test_scaled)
y_pred_prob_test=1/(1+np.exp(-y_pred_test))

model.fit(X_train_scaled,y_pred_prob_train)

joblib.dump(scaler,'scaler_regression.pkl')
joblib.dump(model,'regressor.pkl')

