import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from skimage.feature import hog
from skimage import exposure
from skimage.transform import resize
from time import time
from sklearn.datasets import fetch_lfw_people

# Record the start time
start_time = time()

# Load the ORL Faces dataset
faces_data = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
X = faces_data.images  # Face images
y = faces_data.target  # Target labels (unique integer IDs)
target_names = faces_data.target_names

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
end_time = time()

# Calculate the runtime
runtime = end_time - start_time
print(f"Runtime: {runtime:.2f} seconds")

# Make predictions on the test set
y_pred = clf.predict(X_test_hog)

# Generate a classification report with precision, recall, F1-score, and support
unique_labels = np.unique(y)
# report = classification_report(y_test, y_pred, labels=unique_labels, target_names=[str(label) for label in unique_labels])
report = classification_report(y_test, y_pred, target_names=target_names)
print("Classification Report:")
print(report)

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(unique_labels))
# plt.xticks(tick_marks, [str(label) for label in unique_labels], rotation=45)
# plt.yticks(tick_marks, [str(label) for label in unique_labels])
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Plot a few example faces from the dataset
plt.figure(figsize=(12, 6))
for i in range(10):  # Display the first 10 faces from the test set
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(50, 37), cmap='gray')  # Adjust dimensions based on the actual data
    plt.title(f"Predicted: {target_names[y_pred[i]]}\nActual: {target_names[y_test[i]]}")
    plt.axis('off')

plt.show()