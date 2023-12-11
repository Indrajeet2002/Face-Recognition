import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

# Load the Olivetti Faces dataset
olivetti_faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = olivetti_faces.data
y = olivetti_faces.target
# target_names = olivetti_faces.target_names

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)

# Measure the start time
start_time = time.time()

# Train the SVM classifier on the training data
svm_classifier.fit(X_train, y_train)

# Measure the end time
end_time = time.time()

# Calculate the runtime
runtime = end_time - start_time

print(f"Training time: {runtime:.2f} seconds")

# Make predictions on the testing data
y_pred = svm_classifier.predict(X_test)

# Generate a classification report
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)

# Generate the confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Plot a few example faces from the dataset
plt.figure(figsize=(12, 6))
for i in range(10):  # Display the first 10 faces from the test set
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
    plt.title(f"Predicted: {y_pred[i]}\nActual: {y_test[i]}")
    plt.axis('off')

plt.show()