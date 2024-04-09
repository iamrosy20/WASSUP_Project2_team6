import numpy as np
from keras.preprocessing import image
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils, models, layers, optimizers

# Load images
images = [image.load_img(p, target_size=(130, 70))   # 780, 420
          for p in glob('data/color_front/white/*png') + glob('data/color_front/pink/*png') + glob('data/color_front/yellow/*png')
          + glob('data/color_front/orange/*png') + glob('data/color_front/brown/*png') + glob('data/color_front/blue/*png')
          + glob('data/color_front/lightgreen/*png') + glob('data/color_front/green/*png') + glob('data/color_front/red/*png')
          + glob('data/color_front/gray/*png') + glob('data/color_front/purple/*png') + glob('data/color_front/cyan/*png')
          + glob('data/color_front/black/*png') + glob('data/color_front/violet/*png') + glob('data/color_front/navy/*png')
          + glob('data/color_front/transparent/*png')]
image_vector = np.asarray([image.img_to_array(img) for img in images])

# Set labels
y = [15] * 9482 + [14] * 3445 + [13] * 3079 + [12] * 2286 + [11] * 1525 + [10] * 1426 + [9] * 1224 + [8] * 993 + [7] * 691 + [6] * 196 + [5] * 129 + [4] * 109 + [3] * 71 + [2] * 63 + [1] * 42 + [0] * 25

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(image_vector, y, test_size=0.20, random_state=42)

# Scale the input data
X_train, X_test = X_train / 255, X_test / 255

# Convert to categorical
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

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

# Evaluate the model
score = model.evaluate(X_test, y_test)
print("Test loss:", score[0])       # 0.3502289652824402
print("Test accuracy:", score[1])   # 0.8933037519454956