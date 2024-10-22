#!/usr/bin/env python

'''TODO: add high-level description of this Python script'''

import random as python_random
import argparse
import numpy as np
import pandas as pd
from keras.models import Sequential # type: ignore #
from keras.layers import Dense, LSTM, Bidirectional # type: ignore
import keras
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from tensorflow.keras.optimizers import SGD, Adagrad, Adadelta, Adam, AdamW, Adamax, Adafactor, RMSprop, Nadam, Lion # type: ignore
import tensorflow as tf

# Make reproducible as much as possible
seed=1234
np.random.seed(seed)
tf.random.set_seed(seed)
python_random.seed(seed)
keras.utils.set_random_seed(seed)

def convert_arg(arg):
    '''Converts arguments to correct type. If string, it converts to None, int or float based on format.
    If not string, no conversion is performed.'''
    if type(arg) is not str:
        return arg

    # Handle the case for None
    if arg == 'None':
        return None

    # Try converting to integer
    try:
        return int(arg)
    except (ValueError, TypeError):
        pass  # Move to the next conversion if it fails

    # Try converting to float
    try:
        return float(arg)
    except (ValueError, TypeError):
        pass  # Move to the next conversion if it fails

    # Return the argument as a string if it can't be converted
    return arg

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='train_data_it.csv', type=str,
                        help="Input file to learn from (default train_data_it.csv)")
    parser.add_argument("-d", "--dev_file", type=str, default='dev_data_it.csv',
                        help="Separate dev set to read in (default dev_data_it.csv)")
    parser.add_argument("-t", "--test_file", type=str, default='test_data_it.csv',
                        help="If added, use trained model to predict on test set")
    parser.add_argument("--verbose", default=1,
                        choices=["auto", "0", "1", "2"],
                        help="Verbosity mode for model training (default auto)")
    
    # Baseline model
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Number of samples per gradient update (default 16)")
    parser.add_argument("--epochs", default=50, type=int,
                        help="Number of epochs to train the model (default 50)")
    parser.add_argument("--learning_rate", default=0.01, type=float,
                        help="The learning rate (default 0.01)")

    # Extra LSTM layers + Dropout
    # first
    parser.add_argument("--hidden1", default=128, type=int,
                        help="Number of neurons in the first hidden LSTM layer (default 128)")
    parser.add_argument("--dropout1", default=0.0, type=float,
                        help="Dropout rate in the first hidden LSTM layer (default 0.0)")
    parser.add_argument("--recurrent_dropout1", default=0.0, type=float,
                        help="Recurrent dropout rate in the first hidden LSTM layer (default 0.0)")
    # second
    parser.add_argument("--hidden2", default=0, type=int,
                        help="Number of neurons in the second hidden LSTM layer (default 0)")
    parser.add_argument("--dropout2", default=0.0, type=float,
                        help="Dropout rate in the second hidden LSTM layer (default 0.0)")
    parser.add_argument("--recurrent_dropout2", default=0.0, type=float,
                        help="Recurrent dropout rate in the second hidden LSTM layer (default 0.0)")
    # third (if hidden2==0, hidden3 is ignored!)
    parser.add_argument("--hidden3", default=0, type=int,
                        help="Number of neurons in the third hidden LSTM layer (default 0)")
    parser.add_argument("--dropout3", default=0.0, type=float,
                        help="Dropout rate in the third hidden LSTM layer (default 0.0)")
    parser.add_argument("--recurrent_dropout3", default=0.0, type=float,
                        help="Recurrent dropout rate in the third hidden LSTM layer (default 0.0)")
    
    parser.add_argument("--optimizer", default="SGD", type=str,
                        choices = ["SGD", "Adagrad", "Adadelta", "Adam", "AdamW", "Adamax", "Adafactor", "RMSprop", "Nadam", "Lion"],
                        help="Optimizer (default SGD)")
    
    # Bidirectional
    parser.add_argument("--bidirectional", action="store_true",
                        help="Make all LSTM layers bidirectional")

    args = parser.parse_args()
    for arg in vars(args):
        value=getattr(args, arg)
        nvalue=convert_arg(value)
        setattr(args, arg, nvalue)

    return args


# Load your dataset
def load_data(file_path):
    data = pd.read_csv(file_path)  # assuming your data is in CSV format
    X = data.drop(columns=["IT_M_Label"])  # Features (remove 'IT_M_Label' or 'NST_M_Label' column)
    y = data["IT_M_Label"]
    return X, y


def add_lstm_layer(bidirectional, units, dropout, recurrent_dropout, return_sequences, model):
    if bidirectional:
        model.add(Bidirectional(LSTM(units=units, dropout=dropout, recurrent_dropout=recurrent_dropout, seed=seed, return_sequences=return_sequences)))
    else:
        model.add(LSTM(units=units, dropout=dropout, recurrent_dropout=recurrent_dropout, seed=seed, return_sequences=return_sequences))
    return model

optimizers={
    "SGD": SGD,
    "Adagrad": Adagrad, 
    "Adadelta": Adadelta,
    "Adam": Adam,
    "AdamW": AdamW,
    "Adamax": Adamax,
    "Adafactor": Adafactor,
    "RMSprop": RMSprop,
    "Nadam": Nadam,
    "Lion": Lion
}

def create_model(input_shape, num_labels, args):
    '''Create the Keras model to use'''
    # TODO: Define settings, you might want to create cmd line args for them
    learning_rate = args.learning_rate
    loss_function = 'categorical_crossentropy'
    optim = optimizers[args.optimizer](learning_rate=learning_rate)


    # Now build the model
    model = Sequential()

    # LSTM layers for numerical data
    model = add_lstm_layer(args.bidirectional, args.hidden1, args.dropout1, args.recurrent_dropout1, args.hidden2!=0, model)
    if args.hidden2 > 0:
        model = add_lstm_layer(args.bidirectional, args.hidden2, args.dropout2, args.recurrent_dropout2, args.hidden3 != 0, model)
        if args.hidden3 > 0:
            model = add_lstm_layer(args.bidirectional, args.hidden3, args.dropout3, args.recurrent_dropout3, False, model)

    # Add final dense layer with softmax for classification
    model.add(Dense(units=num_labels, activation="softmax"))
    
    # Compile the model
    optim = optimizers[args.optimizer](learning_rate=args.learning_rate)
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])

    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev, args):
    '''Train the model here. Note the different settings you can experiment with!'''
    # Potentially change these to cmd line args again
    # And yes, don't be afraid to experiment!
    verbose = args.verbose
    batch_size = args.batch_size
    epochs = args.epochs
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor="loss"
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    # Finally fit the model to our data
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[callback], batch_size=batch_size, validation_data=(X_dev, Y_dev))
    # Print final accuracy for the model (clearer overview)
    test_set_predict(model, X_dev, Y_dev, "dev")
    return model


def test_set_predict(model, X_test, Y_test, ident):
    '''Do predictions and measure accuracy on our own test set (that we split off train)'''
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_test, axis=1)
    print('Accuracy on own {1} set: {0}'.format(round(accuracy_score(Y_test, Y_pred), 3), ident))


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()

    # Read in the data and embeddings
    X_train, Y_train = load_data(args.train_file)
    X_dev, Y_dev = load_data(args.dev_file)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)

    # Add a time step dimension (reshape to 3D for LSTM)
    X_train_scaled = np.expand_dims(X_train_scaled, axis=1)
    X_dev_scaled = np.expand_dims(X_dev_scaled, axis=1)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.transform(Y_dev)  # Use transform here

    model = create_model(X_train_scaled.shape[1], len(set(Y_train)), args)
    model = train_model(model, X_train_scaled, Y_train_bin, X_dev_scaled, Y_dev_bin, args)

    # Do predictions on specified test set
    if args.test_file:
        # Read in test set and vectorize
        X_test, Y_test = load_data(args.test_file)

        # Scale and reshape test data
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = np.expand_dims(X_test_scaled, axis=1)  # Reshape to 3D

        # Finally do the predictions
        Y_test_bin = encoder.transform(Y_test)  # Transform test labels to one-hot
        test_set_predict(model, X_test_scaled, Y_test_bin, "test")


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    main()
