

import pandas as pd
import tensorflow as tf
import json

paths = {
    "data": "data/imdb.csv",
    "export-vocab": "exports/vocabulary.json",
    "export-tfjs_model": "exports/model_tfjs"
}
df = pd.read_csv(paths["data"])

BATCH_SIZE = 64
TRAIN_VALID_SPLIT = (.7, .15)
SEED = 42

# Hyperparameters
MAX_VOCAB_SIZE = 30000
MAX_DOC_SIZE = 100
EMBEDDING_DIM = 128

def split_ds(tokenizer, ds_size, texts, labels, Tv_split, batch_size=BATCH_SIZE, shuffle=True):

    fullset = _texts_to_ds(tokenizer, texts, labels)
    if shuffle:
        fullset.shuffle(SEED)

    # Resolves train-valdation-test splits
    # (test split is inferred)
    T_split, v_split = Tv_split

    train_size = int(ds_size * T_split)
    valid_size = int(ds_size * v_split)

    train_ds = fullset.take(train_size)
    remain_ds = fullset.skip(train_size)
    valid_ds = remain_ds.take(valid_size)
    test_ds = remain_ds.skip(valid_size)


    train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    valid_ds = valid_ds.cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, valid_ds, test_ds
    


def _texts_to_ds(tokenizer, texts, labels):
    """
        Turns a collection of text into a TF Dataset object.
        PARAMS:
            texts: Iterable with text strings
            labels: Iterable with text labels
    """
    tokens = tokenizer.texts_to_sequences(texts)
    tokens = tf.keras.utils.pad_sequences(tokens, maxlen=MAX_DOC_SIZE)

    return tf.data.Dataset.from_tensor_slices((tokens, labels))
    
    #return tf.data.Dataset.from_tensor_slices((texts, labels)) \
    #    .cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

def export_vocab(vocabulary, path, max_terms=None):
    """
        Exports vocabulary to path
        PARAMS:
            vocabulary: Dictionary. keys = terms, values = indices
            path: Output path as string
            max_terms: If set, exported vocabulary only includes the top max_term terms
    """
    if max_terms is not None:
        reduced_vocab = {}

        for t, t_i in vocabulary.items():
            if t_i <= max_terms: # keras Tokenizer term indices starts at 1
                reduced_vocab[t] = t_i

        vocabulary = reduced_vocab

    with open(path, 'w') as f:
        json.dump(vocabulary, f)


# Fits naive tokenizer & exports it
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df.text)

ds_size = len(df.text.values)

T, v, t = split_ds(tokenizer, ds_size, df.text.values, df.label.values, Tv_split=TRAIN_VALID_SPLIT)

"""
train_examples_batch, train_labels_batch = next(iter(T.batch(10)))
print(train_examples_batch)
print(train_labels_batch)
"""



# A integer input for vocab indices.
inputs = tf.keras.Input(shape=(None,), dtype="int64")

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

epochs = 3
model.fit(T, validation_data=v, epochs=epochs)
model.evaluate(t)

exit()
print(tf_ds)
print(tf_ds.element_spec[0].shape)


exit()
minids = tf_ds.take(100)
print(type(tf_ds))
print(type(minids))
print(minids)

exit()

export_vocab(tokenizer.word_index, paths["export-vocab"], max_terms=MAX_VOCAB_SIZE)

tokens = tokenizer.texts_to_sequences([df.text[0]])
tokens = tf.keras.utils.pad_sequences(tokens, maxlen=100)
t_wi = tokenizer.word_index
t_wi_ri = list(t_wi.keys())[42]
print(type(t_wi))
print(t_wi_ri)
print(t_wi[t_wi_ri])