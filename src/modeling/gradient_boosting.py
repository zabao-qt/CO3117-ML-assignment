from sklearn.ensemble import GradientBoostingClassifier

def get_model(random_state=42):
  return GradientBoostingClassifier(random_state=random_state)

def get_hyperparameters():
    return {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5]
    }