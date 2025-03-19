# model.py
from keras import models, layers
from keras.callbacks import EarlyStopping

def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))  # Prédiction d'une valeur continue
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def train_model(x_train, y_train, x_val, y_val, epochs=100):
    model = build_model(x_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=16, 
                        validation_data=(x_val, y_val), verbose=0, callbacks=[early_stopping])
    return model, history

def evaluate_model(model, x_test, y_test):
    loss, mae = model.evaluate(x_test, y_test, verbose=0)
    return loss, mae

def predict(model, x_test):
    predictions = model.predict(x_test).flatten()
    return predictions
