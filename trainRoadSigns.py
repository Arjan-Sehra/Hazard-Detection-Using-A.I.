from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(600, 600, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

adam_optimizer = Adam(learning_rate=0.0001)

# Compile the model
model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

base_dir = os.path.dirname('trainRoadSigns.py')
training_images_dir = os.path.join(base_dir, 'trainRoadSigns')

# Prepare your data
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    training_images_dir,  
    target_size=(600, 600), 
    batch_size=16,
    class_mode='binary')

# Train the model
model.fit(train_generator, steps_per_epoch=18, epochs=25)

# Save the model
model.save('road_sign_cnn_model.h5')
