import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import time

# Load the Olivetti faces dataset
faces = fetch_olivetti_faces()
X = faces.data
y = faces.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Perform PCA
start_time = time.time()
n_components = 150
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
end_time_pca = time.time()

# Train a Support Vector Classification (SVC) model
clf = SVC(kernel='rbf', class_weight='balanced')
start_time_training = time.time()
clf.fit(X_train_pca, y_train)
end_time_training = time.time()

# Predict on the test data
start_time_prediction = time.time()
y_pred = clf.predict(X_test_pca)
end_time_prediction = time.time()

# Generate classification report
print(classification_report(y_test, y_pred, zero_division=1))  # Set zero_division to 1 to avoid the warning

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Plot a few example faces from the dataset
plt.figure(figsize=(12, 6))
for i in range(10):  # Display the first 10 faces from the test set
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
    plt.title(f"Predicted: {y_pred[i]}\nActual: {y_test[i]}")
    plt.axis('off')

plt.show()

# Display runtimes
print(f"PCA runtime: {end_time_pca - start_time} seconds")
print(f"Training runtime: {end_time_training - start_time_training} seconds")
print(f"Prediction runtime: {end_time_prediction - start_time_prediction} seconds")
