import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.datasets import fetch_olivetti_faces
from skimage.feature import hog
from skimage import exposure
from skimage.transform import resize
import time

# Record the start time
start_time = time.time()

# Load the Olivetti Faces dataset
faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces_data.images  # Face images
y = faces_data.target  # Target labels (unique integer IDs)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Resize the images to a consistent size for HOG feature extraction
image_size = (64, 64)
X_train_resized = [resize(image, image_size) for image in X_train]
X_test_resized = [resize(image, image_size) for image in X_test]

# Extract HOG features from the resized images
def extract_hog_features(images):
    hog_features = []
    for image in images:
        fd, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hog_features.append(fd)
    return np.array(hog_features)

X_train_hog = extract_hog_features(X_train_resized)
X_test_hog = extract_hog_features(X_test_resized)

# Train a Support Vector Machine (SVM) classifier
clf = SVC(kernel='linear', C=1)
clf.fit(X_train_hog, y_train)

# Record the end time
end_time = time.time()

# Calculate the runtime
runtime = end_time - start_time
print(f"Runtime: {runtime:.2f} seconds")

# Make predictions on the test set
y_pred = clf.predict(X_test_hog)

# Generate a classification report with precision, recall, F1-score, and support
unique_labels = np.unique(y)
report = classification_report(y_test, y_pred, labels=unique_labels, target_names=[str(label) for label in unique_labels])
print("Classification Report:")
print(report)

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_labels)
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(unique_labels))
plt.xticks(tick_marks, [str(label) for label in unique_labels], rotation=45)
plt.yticks(tick_marks, [str(label) for label in unique_labels])

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Visualize the decision boundary in 2D (select two features)
feature1 = 0  # Change this to the feature index you want to plot on the x-axis
feature2 = 1  # Change this to the feature index you want to plot on the y-axis

X_train_2d = X_train_hog[:, [feature1, feature2]]
X_test_2d = X_test_hog[:, [feature1, feature2]]

# Create a mesh grid for the decision boundary plot
xx, yy = np.meshgrid(np.linspace(X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1, 100),
                     np.linspace(X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1, 100))
                     
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu, alpha=0.8)
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange', alpha=0.3)

# Plot data points
scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap=plt.cm.Paired)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("Decision Boundary Visualization")
plt.xlabel(f"Feature {feature1}")
plt.ylabel(f"Feature {feature2}")
plt.show()

# Plot a few example faces from the dataset
plt.figure(figsize=(12, 6))
for i in range(10):  # Display the first 10 faces from the test set
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
    plt.title(f"Predicted: {y_pred[i]}\nActual: {y_test[i]}")
    plt.axis('off')

plt.show()