from tensorflow.python.keras.optimizers import Adam


def train_fitting_room_model(model, train_data, valid_data, epochs=10):
    optimizer = Adam(lr=0.001)
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    history = model.fit(train_data, epochs=epochs, validation_data=valid_data)
    return history
