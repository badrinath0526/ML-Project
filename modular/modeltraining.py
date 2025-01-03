from imblearn.pipeline import Pipeline
from imblearn.over_sampling import KMeansSMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,roc_auc_score,roc_curve,accuracy_score
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import joblib
from Config.configuration import log_info,log_error

#Sequence of steps encapsulated into a single object to automate ML workflow
def build_pipeline():
    return Pipeline([

        ('smote', KMeansSMOTE(sampling_strategy='auto', random_state=22)),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(criterion='entropy', random_state=22))
    ])

#Parameters grid for hyperparameter tuning for each model
def grid_search_tuning(pipeline, X_train, y_train):
    #Random forest
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__class_weight': ['balanced', {0: 1, 1: 2}],
    }


    kfold = StratifiedKFold(shuffle=True, n_splits=2, random_state=22)
    grid_search = GridSearchCV(pipeline, param_grid, scoring='precision',cv=kfold, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    smote = grid_search.best_estimator_.named_steps['smote']
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"Shape of dataset after oversampling: {X_resampled.shape}, {y_resampled.shape}")
    log_info("Hyperparameter tuning is completed after grid search finds the best parameters")
    return grid_search


def calculate_training_accuracy(grid_search, X_train, y_train):
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    log_info("Training accuracy is calculated")
    return train_accuracy

def evaluate_model(grid_search, X_test, y_test,threshold=0.25):
    best_model = grid_search.best_estimator_
    y_prob = best_model.predict_proba(X_test)
    # y_pred = (y_prob >= 0.5).astype(int)
    y_pred = (y_prob[:, 1] >= threshold).astype(int)
    
    report_best = classification_report(y_test, y_pred)
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation precision score: {grid_search.best_score_:.4f}")
    print(report_best)
    log_info("Model is evaluated with best parameters")


# Calculates feature importance score when model is Random Forest
def get_feature_importance(grid_search):
    best_model = grid_search.best_estimator_
    
    # Check if RandomForestClassifier is the model
    if isinstance(best_model.named_steps['model'], RandomForestClassifier):
        feature_importance = best_model.named_steps['model'].feature_importances_
        print("Feature Importance Scores:")
        for idx, score in enumerate(feature_importance):
            print(f"Feature {idx + 1}: {score:.4f}")
    else:
        print("Model is not Random Forest. Feature importance is not available.")

    plt.show()

# Saves the best model to a file using joblib
def save_model(grid_search):
    
    joblib.dump(grid_search.best_estimator_, 'model1.pkl')
    print("Model saved as model1.pkl")


#Performs recursive feature elimination
def perform_rfe(X_train, y_train, estimator=None):
    if estimator is None:
        estimator = RandomForestClassifier(random_state=22)  # You can use any estimator here
    
    rfe = RFE(estimator, n_features_to_select=4)  # Select top 4 features
    rfe.fit(X_train, y_train)
    
    selected_features = X_train.columns[rfe.support_]
    print("Selected Features by RFE:")
    print(selected_features)   


def evaluate_roc_auc(model, X_test, y_test):
    
    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC Score: {roc_auc:.4f}")


    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()