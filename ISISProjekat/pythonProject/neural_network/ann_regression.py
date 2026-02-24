import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class AnnRegression:
    def __init__(self):
        self.model = None

        # Hiperparametri - podrazumevane vrednosti
        #
        # Do sada mi je najbolje bilo 3,76% do 4,19% - 5 slojeva. 20 neurona, 100 epoha
        #                                                    4,40% - 5, 18, 100
        self.number_of_hidden_layers = 5
        self.number_of_neurons_in_first_hidden_layer = 20
        self.number_of_neurons_in_other_hidden_layers = 20
        self.epoch_number = 100
        self.batch_size_number = 32

        self.activation_function = "relu"

        self.loss_function = "mean_absolute_error"
        self.optimizer = Adam(learning_rate=0.001)

        # Regularizacija, gasi 20% neurona nasumicno da spreci overfitting
        self.dropout_rate = 0.2

    def build_model(self, input_dim):
        model = Sequential()

        # Prvi skriveni sloj
        model.add(Dense(
            self.number_of_neurons_in_first_hidden_layer,
            activation=self.activation_function,
            input_dim=input_dim
        ))
        model.add(Dropout(self.dropout_rate))

        # Ostali skriveni slojevi
        for i in range(self.number_of_hidden_layers - 1):
            model.add(Dense(
                self.number_of_neurons_in_other_hidden_layers,
                activation=self.activation_function
            ))
            model.add(Dropout(self.dropout_rate))

        # Izlazni sloj (regresija)
        model.add(Dense(1, activation="linear"))

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=["mae"]
        )

        self.model = model

    def compile_and_fit(self, X, y):
        if self.model is None:
            self.build_model(X.shape[1])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Ogranicenje da load ne bude manji od 0 ili previse velik
        y = np.clip(y, 0, np.max(y) * 1.5)

        # Ako se loss ne poboljsa 5 epoha vracam najbolje tezine
        early_stop = EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True
        )

        print(f"Pocinjem trening (max {self.epoch_number} epoha)...")

        history = self.model.fit(
            X,
            y,
            epochs=self.epoch_number,
            batch_size=self.batch_size_number,
            callbacks=[early_stop],
            verbose=1 #progress bar i loss
        )

        print("Trening zavrsen")
        return history

    def save_model(self, path):
        if self.model is None:
            raise Exception("Nema modela za cuvanje")
        self.model.save(path)
        print(f"Model sacuvan u fajl: {path}")

    def load_model(self, path):
        if not os.path.exists(path):
            raise Exception(f"Model fajl {path} ne postoji")
        self.model = load_model(path, compile=True)
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=["mae"]
        )
        print(f"Model ucitan iz fajla: {path}")

    def predict(self, X):
        if self.model is None:
            raise Exception("Model nije ucitan")

        X = np.array(X, dtype=np.float32)
        predictions = self.model.predict(X, verbose=0)

        return predictions.flatten()
