import numpy as np
from keras.preprocessing import image
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils, models, layers, optimizers

# Load images
images = [image.load_img(p, target_size=(130, 70))   # 780, 420
          for p in glob('data/color_back/white/*png') + glob('data/color_back/pink/*png') + glob('data/color_back/yellow/*png')
          + glob('data/color_back/orange/*png') + glob('data/color_back/brown/*png') + glob('data/color_back/blue/*png')
          + glob('data/color_back/lightgreen/*png') + glob('data/color_back/green/*png') + glob('data/color_back/red/*png')
          + glob('data/color_back/gray/*png') + glob('data/color_back/purple/*png') + glob('data/color_back/cyan/*png')
          + glob('data/color_back/black/*png') + glob('data/color_back/violet/*png') + glob('data/color_back/navy/*png')
          + glob('data/color_back/transparent/*png')]
image_vector = np.asarray([image.img_to_array(img) for img in images])

# Set labels
y = [15] * 10288 + [14] * 3528 + [13] * 3429 + [12] * 2166 + [11] * 1332 + [10] * 1058 + [9] * 1205 + [8] * 696 + [7] * 553 + [6] * 219 + [5] * 62 + [4] * 87 + [3] * 65 + [2] * 48 + [1] * 5 + [0] * 45

# Split the dataset
X_train = image_vector
y_train = y

# Scale the input data
X_train = X_train / 255

# Convert to categorical
y_train = utils.to_categorical(y_train)

# Build model
def build(input_shape, classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))
    return model

# Initialize and fit the model
model = build((130, 70, 3), 16)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=0.001), metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

# Save the model
model.save('cnn_color_right_final.h5')