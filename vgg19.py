# import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import Sequential
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import numpy as np



# prepare a list of image files to be loaded
def image_files(input_directory):
    filepaths = []
    labels = []

    digit_folders = os.listdir(input_directory)
    # print(digit_folders)

    for digit in digit_folders:
        path = os.path.join(input_directory, digit)
        flist = os.listdir(path)
        for f in flist:
            fpath = os.path.join(path, f)
            filepaths.append(fpath)
            labels.append(digit)
    return filepaths, labels


def load_images(filepaths):
    images = []
    for i in tqdm(range(len(filepaths))):
        img = load_img(filepaths[i], target_size=(32, 32, 3), grayscale=False)
        img = img_to_array(img)
        img.astype('float32')
        img = img / 255
        images.append(img)

    images = np.array(images)
    return images


# load the paths and labels in differnt variables
directory_10k = r'C:\Users\User\PycharmProjects\vgg\TRAIN'
filepaths, labels = image_files(directory_10k)

# load the 10K images
images = load_images(filepaths)

y = to_categorical(labels, num_classes=28)
X_train, X_test, y_train, y_test = train_test_split(images, y, random_state=42, test_size=0.2)

print(X_train.shape)
print(X_test.shape)

vgg19 = VGG19(weights='imagenet',
              include_top=False,
              input_shape=(32, 32, 3)
              )

model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(28, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train,
                    epochs=14,
                    batch_size=128,
                    validation_data=(X_test, y_test)
                    )

score = model.evaluate(X_test, y_test)
print(score)

# Save the trained model
model.save('my_model.h5')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

axes[0].plot(history.history['accuracy'])
axes[0].plot(history.history['val_accuracy'])
axes[0].set_title('Model Accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['train', 'validation'], loc='upper left')

axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model Loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['train', 'validation'], loc='upper left')

plt.show()

# # encode the letter names
# label_encoder = LabelEncoder()
# labels_encoded = label_encoder.fit_transform(labels)
#
# model = load_model('my_model.h5')
#
# img_path = 'yud.png'
# img = Image.open(img_path)
# img = img.resize((32, 32))
# img = np.array(img)
# img = img / 255.0
# img = np.expand_dims(img, axis=0)
#
# pred = model.predict(img)
# max_class = np.argmax(pred)
# # letter_name = label_encoder.inverse_transform([np.argmax(pred)])
# print("the letter is:", max_class)
#
# # evaluate the model on your test data
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
#
# print('Test loss:', test_loss)
# print('Test accuracy:', test_accuracy)
