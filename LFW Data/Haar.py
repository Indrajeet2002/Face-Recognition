import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import itertools
from sklearn.datasets import fetch_lfw_people

# Function to detect faces using Haar Cascades
def detect_face_haar_cascades(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces, gray

# Load the LFW dataset and perform face recognition
def face_recognition_lfw():
    start_time = time.time()

    lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Train a Support Vector Machine (SVM) classifier
    clf = SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train_encoded)

    # Record the runtime
    end_time = time.time()

    # Calculate the runtime
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

    # Make predictions
    y_pred = clf.predict(X_test)

    # Generate classification report
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
        plt.imshow(X_test[i].reshape(50, 37), cmap='gray')  # Adjust dimensions based on the actual data
        plt.title(f"Predicted: {target_names[y_pred[i]]}\nActual: {target_names[y_test[i]]}")
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    face_recognition_lfw()
