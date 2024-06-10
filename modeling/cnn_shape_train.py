import numpy as np
from tensorflow.keras.preprocessing import image
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils, models, layers, optimizers

# Load images
images = [image.load_img(p, target_size=(130, 70))   # 780, 420
          for p in glob('data/shape/circle/*png') + glob('data/shape/rectangle/*png') + glob('data/shape/ellipse/*png') + glob('data/shape/square/*png')
          + glob('data/shape/octagon/*png') + glob('data/shape/triangle/*png') + glob('data/shape/rhombus/*png') + glob('data/shape/pentagon/*png')
          + glob('data/shape/hexagon/*png') + glob('data/shape/semicircle/*png') + glob('data/shape/etc/*png')]
image_vector = np.asarray([image.img_to_array(img) for img in images])

# Set labels
y = [10] * 9706 + [9] * 6948 + [8] * 6666 + [7] * 276 + [6] * 274 + [5] * 235 + [4] * 91 + [3] * 58 + [2] * 50 + [1] * 3 + [0] * 479

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
    model.add(layers.Conv2D(128, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))
    return model

# Initialize and fit the model
model = build((130, 70, 3), 11)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(learning_rate=0.001), metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

# Save the model
model.save('cnn_shape_final.h5')