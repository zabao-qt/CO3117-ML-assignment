
# 📚 Machine Learning Models - Student Performance Classification

## 📂 Project Structure
All source code is organized under the `src/modeling` folder:
- `Data_loader.py`: Load and preprocess dataset.
- `decision_tree.py`: Train and test Decision Tree model.
- `gradient_boosting.py`: Train and test Gradient Boosting model.
- `knn.py`: Train and test K-Nearest Neighbors (KNN) model.
- `logistic_regression.py`: Train and test Logistic Regression model.
- `random_forest.py`: Train and test Random Forest model.
- `svm.py`: Train and test Support Vector Machine (SVM) model.
- `train.py`: Main script to run experiments on KNN and Gradient Boosting with multiple hyperparameters.

---

## ▶️ How to Run

1. **Run experiments on KNN and Gradient Boosting (more detailed testing):**
   ```bash
   cd src/modeling
   python train.py
   ```

2. **Quickly test individual models (using default hyperparameters):**
   ```bash
   python logistic_regression.py
   python decision_tree.py
   python random_forest.py
   python svm.py
   ```

> ✅ You can run each `.py` file independently. No special configuration needed.

---

## 📦 Requirements

- Python 3.8+
- Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📄 Notes
- The dataset is automatically loaded inside `Data_loader.py`.
- Training results (accuracy, classification report, confusion matrix, etc.) are printed to the console.
- For KNN and Gradient Boosting, `train.py` tests different hyperparameter values and prints comparison results.

---

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── Math-Student.csv 
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks 
│   ├── Data_understanding    <- Jupyter notebooks. 
│
│
├── src
    └── modeling
        ├── Data_loader.py          <- Data loading and preprocessing
        ├── decision_tree.py         <- Train Decision Tree model
        ├── gradient_boosting.py     <- Train Gradient Boosting model
        ├── knn.py                   <- Train K-Nearest Neighbors model
        ├── logistic_regression.py   <- Train Logistic Regression model
        ├── random_forest.py         <- Train Random Forest model
        ├── svm.py                   <- Train Support Vector Machine model
        └── train.py                 <- Main script for KNN and Gradient Boosting
```

--------