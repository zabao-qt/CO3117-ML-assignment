# src/modeling/train.py

import numpy as np 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
import importlib
import sys
import os

# Import the data loader 
try:
    from Data_loader import load_data
except ImportError:
    print("Error: Could not import 'load_data' from Data_loader.py.")
    print("Ensure Data_loader.py is in the same directory as train.py (src/modeling).")
    sys.exit(1)

# Model Selection 
# Default model. Change this string to the filename (without .py) of the model you want to run.
DEFAULT_MODEL_NAME = "gradient_boosting"

if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]
else:
    MODEL_NAME = DEFAULT_MODEL_NAME
    print(f"No model specified via command line. Using default: {MODEL_NAME}")

# Dynamic Model Loading 
model = None
hyperparameters = None
model_display_name = MODEL_NAME # Initial display name

try:
    # Adjust path for importing modules from the parent directory's 'models' folder
    model_module_path = f"..modeling.{MODEL_NAME}" 
    model_file_path = os.path.join('..', 'modeling', f'{MODEL_NAME}.py')
    script_dir = os.path.dirname(__file__)
    absolute_model_file_path = os.path.abspath(os.path.join(script_dir, model_file_path))
    if not os.path.exists(absolute_model_file_path):
         raise FileNotFoundError(f"Model file not found: {absolute_model_file_path}")

    try:
        model_module = importlib.import_module(f"modeling.{MODEL_NAME}", package='src')
    except ModuleNotFoundError:
        src_dir = os.path.abspath(os.path.join(script_dir, '..'))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        try:
             model_module = importlib.import_module(f"modeling.{MODEL_NAME}")
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Could not find the model module src.modeling.{MODEL_NAME}. Ensure __init__.py exists in src/modeling.")


    # Get the model instance
    if hasattr(model_module, 'get_model'):
        get_model_func = getattr(model_module, 'get_model')
        model = get_model_func() # Instantiate the model
        print(f"Successfully loaded model: {MODEL_NAME} ({type(model).__name__})")
    else:
        raise AttributeError(f"Module '{model_module_path}' does not have a 'get_model' function.")

    # Optionally, check for hyperparameters
    if hasattr(model_module, 'get_hyperparameters'):
        get_hyperparameters_func = getattr(model_module, 'get_hyperparameters')
        hyperparameters = get_hyperparameters_func()
        print(f"Found hyperparameters for {MODEL_NAME}.")

except ModuleNotFoundError as e:
    print(f"Error: Could not find the model module. Details: {e}")
    print("Ensure:")
    print(f"  1. The file 'src/modeling/{MODEL_NAME}.py' exists.")
    print(f"  2. The 'src/modeling' folder contains an '__init__.py' file (can be empty).")
    print(f"  3. You are running this script correctly (e.g., from the 'src' directory: python modeling/train.py)")
    sys.exit(1)
except FileNotFoundError as e:
    print(e)
    sys.exit(1)
except AttributeError as e:
    print(f"Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    sys.exit(1)


# Setup
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# Data Loading
data_file_path = '../../data/Math-Students.csv'
print(f"\nLoading and preprocessing data using Data_loader from: {data_file_path}")

try:
    # Use the imported load_data function
    train_dataset, test_dataset = load_data(data_file_path, train_split=0.8)

    # Extract data for scikit-learn models
    # train_dataset.tensors gives a tuple (X_train_tensor, y_train_tensor)
    X_train_tensor, y_train_tensor = train_dataset.tensors
    X_test_tensor, y_test_tensor = test_dataset.tensors

    # Convert tensors to numpy arrays for scikit-learn
    X_train_processed = X_train_tensor.numpy()
    y_train = y_train_tensor.numpy()
    X_test_processed = X_test_tensor.numpy()
    y_test = y_test_tensor.numpy()

    print(f"Data loaded and preprocessed via Data_loader.")
    print(f"Training data shape: {X_train_processed.shape}")
    print(f"Testing data shape: {X_test_processed.shape}")
    print(f"Target distribution in training set (0=Fail, 1=Pass): {np.bincount(y_train.astype(int))}")
    print(f"Target distribution in test set (0=Fail, 1=Pass): {np.bincount(y_test.astype(int))}")

    # Model Training and Evaluation
    print(f"\nTraining and Evaluating {model_display_name}")

    model_to_evaluate = model # Start with the base model

    # Perform hyperparameter tuning if parameters are available
    if hyperparameters and isinstance(hyperparameters, dict) and hyperparameters:
        print(f"Performing GridSearchCV for {model_display_name}...")
        try:
            # Use the base model instance for GridSearchCV
            grid_search = GridSearchCV(estimator=model, param_grid=hyperparameters, cv=5, scoring='accuracy', n_jobs=-1, error_score='raise')
            grid_search.fit(X_train_processed, y_train) 

            print(f"Best parameters found: {grid_search.best_params_}")
            print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
            model_to_evaluate = grid_search.best_estimator_
            model_display_name = f"{MODEL_NAME} (Best GridSearch)"
        except Exception as e:
            print(f"GridSearchCV failed: {e}. Training with default parameters instead.")
            model.fit(X_train_processed, y_train)
            print(f"{model_display_name} model trained directly after GridSearchCV error.")
    else:
        # Train the model directly if no hyperparameters
        model.fit(X_train_processed, y_train) 
        print(f"{model_display_name} model trained directly.")

    # Make predictions using the chosen model
    y_pred = model_to_evaluate.predict(X_test_processed)

    # Evaluate the model
    acc = accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"\n🔍 {model_display_name} Test Accuracy: {acc:.2f}%")
    print("Classification Report:\n", report)

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail (0)", "Pass (1)"])
    disp.plot(cmap='Blues')
    plt.title(f'{model_display_name} Confusion Matrix')
    plt.show()

except FileNotFoundError:
    print(f"Error: The data file '{data_file_path}' was not found.")
    print("Please make sure it's in the correct directory relative to train.py.")
except Exception as e:
    print(f"An error occurred during data processing or model evaluation: {e}")

print(f"\n{model_display_name} Evaluation Complete")