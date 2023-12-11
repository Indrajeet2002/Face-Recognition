import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

# Load the LFW (Labeled Faces in the Wild) dataset
lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.4)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(lfw_people.data, lfw_people.target, test_size=0.2, random_state=42)
target_names = lfw_people.target_names

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
classification_rep = classification_report(y_test, y_pred, target_names=target_names)

# Save the classification report to a text file
# with open("classification_report.txt", "w") as text_file:
print("Classification Report:", classification_rep)
    # print(classification_rep, file=text_file)

# Display the confusion matrix and sample faces as before
confusion_mat = confusion_matrix(y_test, y_pred, labels=range(len(target_names)))

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
    plt.imshow(X_test[i].reshape(50, 37), cmap='gray')  # Adjust dimensions based on the actual data
    plt.title(f"Predicted: {target_names[y_pred[i]]}\nActual: {target_names[y_test[i]]}")
    plt.axis('off')

plt.show()
