import random

import numpy as np
from keras.utils import np_utils
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# make result reproducible
seed = 2022
random.seed(seed)
np.random.seed(seed)

# data preprocessing
X, y = load_iris(return_X_y=True)
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y_onehot = np_utils.to_categorical(y)

# construct model
model = Sequential()
model.add(Dense(6, activation="leaky_relu", input_dim=X.shape[1]))
model.add(Dense(3, activation="softmax"))
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# set fitting parameter
stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode="min", restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=10, verbose=1, mode="min", min_lr=1e-5, factor=0.5)
tensorboad = TensorBoard(log_dir="./test", write_graph=True, write_images=True)

# model train and valid
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
i = 0
for train_i, test_i in kfold.split(X, y_onehot):
    print(f"----- the {i} cross validation -----")
    train_x, train_y = X[train_i], y_onehot[train_i]
    test_x, test_y = X[test_i], y_onehot[test_i]
    history = model.fit(
        train_x,
        train_y,
        batch_size=15,
        epochs=200,
        verbose=1,
        callbacks=[reduce_lr, tensorboad, stopping],
        validation_data=[test_x, test_y],
        shuffle=True
    )
    i += 1
