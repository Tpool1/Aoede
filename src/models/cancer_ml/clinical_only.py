from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import MeanIoU

class clinical_only():

    def __init__(self, load_model=True):
        self.load_model = load_model
    
    def train_model(self, X_train, y_train, epochs=10, batch_size=32):
        input = keras.layers.Input(shape=(self.X_train.shape[1],))

        x = Dense(9, activation=self.activation_function)(input)
        x = Dense(9, activation=self.activation_function)(x)
        x = Dense(6, activation=self.activation_function)(x)
        x = Dense(4, activation=self.activation_function)(x)
        x = Dense(2, activation=self.activation_function)(x)
        output = Dense(1, activation='linear')(x)
        model = keras.Model(input, output)

        model.compile(optimizer='SGD',
                            loss='mean_squared_error',
                            metrics=['accuracy', MeanIoU(num_classes=2)])

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        model.save('data\\saved_models\\text_prediction\\keras_clinical_only_model.h5')

        return model

    def get_model(self, X_train=None, y_train=None, epochs=None, batch_size=None):
        
        if self.load_model:
            self.model = keras.models.load_model('data\\saved_models\\text_prediction\\keras_clinical_only_model.h5')
        else:
            self.model = self.train_model(X_train, y_train, epochs, batch_size)

        return self.model
