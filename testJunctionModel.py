from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import os

base_dir = os.path.dirname('testJunctionModel.py')

load_model_dir = os.path.join(base_dir, 'junction_cnn_model.h5')
testing_images_dir = os.path.join(base_dir, 'testingJunctionImages')

model = load_model(load_model_dir)

### Testing against real dashcam footage ###
test_datagen_2 = ImageDataGenerator(rescale=1./255)
dashcam_footage = test_datagen_2.flow_from_directory(
    testing_images_dir,  
    target_size=(600,600),
    batch_size=30,
    class_mode='binary',
    shuffle=False)

loss, accuracy = model.evaluate(dashcam_footage, steps=1)
print(f"Test Accuracy: {accuracy * 100}%")

predictions = model.predict(dashcam_footage, steps=1)
predicted_classes = (predictions > 0.5).astype("int32")

true_classes = dashcam_footage.classes

tp, fn, fp, tn = confusion_matrix(true_classes, predicted_classes.flatten()).ravel()
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Negatives: {tn}")

recall = tp / (tp + fn)
print(f"Recall score: {recall}")

precision = tp / (tp + fp)
print(f"Precision score: {precision}")

f1_score = 2*((precision*recall) / (precision+recall))
print(f"F1-Score: {f1_score}")


filenames = dashcam_footage.filenames

for i in range(len(filenames)):
    actual_label = true_classes[i]
    predicted_label = predicted_classes[i][0]
    filename = filenames[i]
    result = "Passed" if actual_label == predicted_label else "failed"
    print(f"Filename: {filename}, Result: {result}")