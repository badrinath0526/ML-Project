from preprocessing import load_and_clean_data, preprocess_features, split_data
from plotting import plot_class_distribution, plot_correlation_matrix, plot_histograms, plot_boxplot,plot_pairplot
from modeltraining import build_pipeline, grid_search_tuning, evaluate_model,save_model,get_feature_importance,calculate_training_accuracy

# Load and preprocess data
df = load_and_clean_data("data/diabetes_prediction_dataset.csv")
X, y = preprocess_features(df)
X_train, X_test, y_train, y_test = split_data(X, y)
# perform_chi_square_test(df,target_col='diabetes')
# small_df=df.iloc[30000:60000]
# Visualization
plot_class_distribution(df)
plot_correlation_matrix(df)
numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
plot_histograms(X_train, numerical_features)
plot_boxplot(df)
# plot_pairplot(small_df,target_column="diabetes")

# Model training and evaluation
pipeline = build_pipeline()
grid_search = grid_search_tuning(pipeline, X_train, y_train)
train_accuracy = calculate_training_accuracy(grid_search, X_train, y_train)
print(f"Training Accuracy: {train_accuracy:.4f}")
get_feature_importance(grid_search)
evaluate_model(grid_search, X_test, y_test)
best_model = grid_search.best_estimator_

# evaluate_roc_auc(best_model, X_test, y_test)
save_model(grid_search)