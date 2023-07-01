# VGG16 Pretrained Model- Cifar10 Data - TensorFlow & Keras Framework- Functionnal API

# import needed libraries
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.config import list_physical_devices
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Check if GPU is available
print("GPU is available" if list_physical_devices('GPU') else "GPU is not available")

# All available devices
list_physical_devices()


# Load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Create train and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


# Verify the shape of each set
X_train.shape, X_val.shape, X_test.shape


# Define label names
label_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


# Select 36 indices at random
indices = np.random.randint(0, len(X_train), size=36)

# 6 x 6 grid
fig, axes = plt.subplots(6, 6, figsize=(8, 8))

# Iterate over the indices and display the corresponding images
for i, ax in enumerate(axes.flat):
    # Index and label of the image
    index = indices[i]
    label = y_train[index][0]  # Get the label value
    
    # Display the image
    ax.imshow(X_train[index])
    
    # Set the title with the label name
    ax.set_title(f"{label_names[label].capitalize()}")
    
    # Remove the grid
    ax.axis('off')

# Show the plot
plt.tight_layout()
plt.show()



# Model

# Define VGG16 model for transfer learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False

# Add a new top layer which corresponds to our own number of classes
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)  # Dropout rate of 20%
predictions = Dense(10, activation='softmax')(x)

# Define the new model
model = Model(inputs=base_model.input, outputs=predictions)

# model summary
model.summary()



# Compile the model
optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping & checkpointing the best model in ../working directory
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta = 0.1, factor=0.9, patience=2, min_lr=0.00005)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)



# Train the model
model_hist = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                       validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr, model_checkpoint])





# Plot train & test set losses
train_loss = model_hist.history['loss']
val_loss = model_hist.history['val_loss']

epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



# Test the model's performance on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)



# Classification report 

# Model's prediction probabilities for each class for the test set
y_pred_probas = model.predict(X_test)

# Convert probability estimates into class predictions by selecting the class with the highest probability
y_pred = np.argmax(y_pred_probas, axis=1)

# Generate a classification report by comparing the model's predictions (y_pred) with the true labels (y_test)
report = classification_report(y_test, y_pred)
print(report)

