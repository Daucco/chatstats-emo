# Simple TF model for binayr classification

from tokenizer import Tokenizer
from dataset import DatasetFromCSV
import tensorflow as tf
import json

BATCH_SIZE = 64
TRAIN_VALID_SPLIT = (.7, .15)
SEED = 42

# Hyperparameters
MAX_VOCAB_SIZE = 30000
MAX_DOC_SIZE = 100
EMBEDDING_DIM = 128
EPOCHS = 3

# A integer input for vocab indices.
inputs = tf.keras.Input(shape=(None,), dtype="int32")

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
x = tf.keras.layers.Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM)(inputs)
x = tf.keras.layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = tf.keras.layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = tf.keras.layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = tf.keras.layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

if __name__ == "__main__":
    tok_path = "exports/tokenizer.json"
    data_path = "data/imdb.csv"
    export_path = "exports/minimodel.h5"

    # Loads tokenizer from local model
    tokenizer = Tokenizer(max_terms=MAX_VOCAB_SIZE)
    tokenizer.load(tok_path)

    # Initializes dataset object (TODO export dataset)
    dds = DatasetFromCSV(
        datapath = data_path,
        doccol = "text",
        labelcol = "label",
        splits = TRAIN_VALID_SPLIT,
        tokenizer = tokenizer,
    )

    # Generates tf datasets
    dds.generate_tf(batch_size=BATCH_SIZE, maxlen=MAX_DOC_SIZE)
    T, v, t = dds.tf_datasets

    # Trains model
    model.fit(T, validation_data=v, epochs=EPOCHS)

    # Evaluates model on test set and exports weights
    model.evaluate(t)
    model.save(export_path)