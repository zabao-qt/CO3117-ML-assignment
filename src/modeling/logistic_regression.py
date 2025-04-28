from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from Data_loader import load_data
import matplotlib.pyplot as plt

def run_logistic_regression():
    train_dataset, test_dataset = load_data('../../data/Math-Students.csv')
    
    X_train, y_train = train_dataset.tensors
    X_test, y_test = test_dataset.tensors

    # Convert tensors to numpy
    X_train = X_train.numpy()
    y_train = y_train.numpy()
    X_test = X_test.numpy()
    y_test = y_test.numpy()

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    print("=== Logistic Regression Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=["Fail", "Pass"])
    disp.plot(cmap='Blues')
    plt.title('Logistic Regression Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    run_logistic_regression()
