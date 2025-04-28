from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from Data_loader import load_data
import matplotlib.pyplot as plt

def run_svc():
    train_dataset, test_dataset = load_data('../../data/Math-Student.csv')

    X_train, y_train = train_dataset.tensors
    X_test, y_test = test_dataset.tensors

    X_train = X_train.numpy()
    y_train = y_train.numpy()
    X_test = X_test.numpy()
    y_test = y_test.numpy()

    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\n {'Support Vector Machine'} Accuracy: {acc:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
    disp.plot(cmap='Blues')
    plt.title(f'{'Support Vector Machine'} Confusion Matrix')
    plt.show()
    
if __name__ == "__main__":
    run_svc()