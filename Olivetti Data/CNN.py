import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam

start_time = time.time()

# Load the Olivetti dataset
olivetti = fetch_olivetti_faces()
x_data, y_data = olivetti.data, olivetti.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1234)

# Normalize the data to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape the data
im_rows, im_cols = 64, 64  # Adjust these dimensions as needed
im_shape = (im_rows, im_cols, 1)

x_train = x_train.reshape(x_train.shape[0], im_rows, im_cols, 1)
x_test = x_test.reshape(x_test.shape[0], im_rows, im_cols, 1)

# Define the number of classes
num_classes = len(np.unique(y_data))

# Convert labels to categorical format
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# Create the CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=im_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
cnn_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Print the summary of the model
cnn_model.summary()

# Fit the model to the training data
history = cnn_model.fit(
    x_train, y_train, 
    batch_size=64,
    epochs=50, 
    verbose=2,
    validation_split=0.1
)



# Evaluate the model on the test data
eval_start_time = time.time()
scores = cnn_model.evaluate(x_test, y_test, verbose=0)
eval_end_time = time.time()
print(f'Test loss: {scores[0]}')
print(f'Test accuracy: {scores[1]}')

# Plot the accuracy and loss curves
plot_start_time = time.time()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plot_end_time = time.time()

# Make predictions on the test data
y_pred = cnn_model.predict(x_test)

# Calculate the accuracy score
# accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
# print("Accuracy:", accuracy)

# Calculate and display the confusion matrix and classification report
cnf_start_time = time.time()
cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
cnf_end_time = time.time()

# Display confusion matrix as heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cnf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Plot a few example faces from the dataset
plt.figure(figsize=(12, 6))
for i in range(10):  # Display the first 10 faces from the test set
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(64, 64), cmap='gray')
    plt.title(f"Predicted: {y_pred[i]}\nActual: {y_test[i]}")
    plt.axis('off')

plt.show()

print('Classification report:')
# print(classification_report(np.argmax(y_test, axis=1), y_pred))
print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), zero_division=1))

end_time = time.time()
print(f"Total runtime: {end_time - start_time} seconds")
print(f"Evaluation runtime: {eval_end_time - eval_start_time} seconds")
print(f"Plotting runtime: {plot_end_time - plot_start_time} seconds")
print(f"Confusion matrix runtime: {cnf_end_time - cnf_start_time} seconds")