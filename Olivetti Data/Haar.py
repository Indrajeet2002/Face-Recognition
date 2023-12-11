import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import itertools

# Function to detect faces using Haar Cascades
def detect_face_haar_cascades(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces, gray

# Load the Olivetti dataset and perform face recognition
def face_recognition_olivetti():
    start_time = time.time()

    # Load the Olivetti Faces dataset
    olivetti = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = olivetti.images
    y = olivetti.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train_flat = [face.flatten() for face in X_train]
    X_test_flat = [face.flatten() for face in X_test]

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Train a Support Vector Machine (SVM) classifier
    clf = SVC(kernel='linear', C=1)
    clf.fit(X_train_flat, y_train_encoded)

    # Record the runtime
    end_time = time.time()

    # Calculate the runtime
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

    # Make predictions
    y_pred = clf.predict(X_test_flat)

    # Generate classification report
    target_names = [str(i) for i in np.unique(y_test)]
    report = classification_report(y_test_encoded, y_pred, target_names=target_names)
    print(report)

    # Display confusion matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # Display model graph (not reshaped)
    plt.figure(figsize=(8, 6))
    plt.title('SVM Classifier Model')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.plot(clf.coef_.T)
    plt.show()

    # Plot a few example faces from the dataset
    plt.figure(figsize=(12, 6))
    for i in range(10):  # Display the first 10 faces from the test set
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
        plt.title(f"Predicted: {y_pred[i]}\nActual: {y_test[i]}")
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    face_recognition_olivetti()

