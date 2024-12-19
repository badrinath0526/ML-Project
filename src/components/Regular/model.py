from data_preprocessing import X_train,y_train,y_test,X_test
from imblearn.pipeline import Pipeline 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import KMeansSMOTE,SMOTE,RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
# import visualization

# param_grid = {
#     'C': [0.1, 1, 10],  # Regularization parameter
#     'solver': ['liblinear', 'saga'],
# }

# grid_search = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=5)
# grid_search.fit(X_train_scaled, y_train)
# best_model = grid_search.best_estimator_
# y_pred_best = best_model.predict(X_test_scaled)


# report_best = classification_report(y_test, y_pred_best)
# print(report_best)
# param_grid = {
#     'model__C': [0.1, 1, 10],  # Regularization parameter
#     'model__kernel': ['linear', 'rbf', 'poly'],  # Different kernel types
#     'model__gamma': ['scale', 'auto'],  # Gamma values
# }
param_grid = {
    'model__n_estimators': [50, 100, 200],  # Number of trees
    'model__max_depth': [None, 10, 20, 30],  # Depth of trees
    'model__min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'model__class_weight': ['balanced', {0: 1, 1: 2}],  # Class weight options
}

pipeline=Pipeline([
('smote',KMeansSMOTE(sampling_strategy='auto',random_state=22)),
 ('scaler',StandardScaler()),
 ('model',RandomForestClassifier(criterion='entropy',random_state=22))
])
kfold=StratifiedKFold(shuffle=True,n_splits=2,random_state=22)
grid_search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring='precision', verbose=1,n_jobs=-1)
# grid_search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring='precision', verbose=1)

# cv_scores=cross_val_score(pipeline,X_train,y_train,cv=kfold,scoring='precision')

# print(f"Cross-validation accuracy scores: {cv_scores}")
# print(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")
# print(f"Standard deviation of cross-validation accuracy: {cv_scores.std():.4f}")

# pipeline.fit(X_train,y_train)

# y_pred=pipeline.predict(X_test)

# report=classification_report(y_test,y_pred)
# print(report)
grid_search.fit(X_train, y_train)

# Print the best parameters and score found
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation precision score: {grid_search.best_score_:.4f}")

# Use the best model from grid search to predict on the test data
y_pred_best = grid_search.best_estimator_
y_prob=y_pred_best.predict_proba(X_test)
threshold=0.2
y_pred_adjusted=(y_prob[:,1]>=threshold).astype(int)
# Print the classification report for the best model
report_best = classification_report(y_test, y_pred_adjusted)
print(report_best)


# grid_search.fit(X_train, y_train)

# # Print the best parameters and score found
# print(f"Best parameters found: {grid_search.best_params_}")
# print(f"Best cross-validation precision score: {grid_search.best_score_:.4f}")

# # Use the best model from grid search to predict on the test data
# y_pred_best = grid_search.best_estimator_.predict(X_test)

# # Print the classification report for the best model
# report_best = classification_report(y_test, y_pred_best)
# print(report_best)

 