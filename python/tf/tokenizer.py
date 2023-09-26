# Simple tokenizer to get tokens from raw text
# This should be replaced with proper NLP ops: cleaning + prepro + vectorizer
# Text operations should match between the python and js app!
 
import tensorflow as tf
import json


class Tokenizer():
    def __init__(self, max_terms=None):
        self.max_terms = max_terms

    def _export_vocab(self, path):
        vocabulary = self._tokenizer.word_index
        max_terms = self.max_terms

        if max_terms is not None:
            reduced_vocab = {}

            for t, t_i in vocabulary.items():
                if t_i <= max_terms: # keras Tokenizer term indices starts at 1
                    reduced_vocab[t] = t_i

            vocabulary = reduced_vocab

        with open(path, 'w') as f:
            json.dump(vocabulary, f)

    def texts_to_sequences(self, texts, labels=None, maxlen=None):
        tokens = self._tokenizer.texts_to_sequences(texts)
        if maxlen is not None:
            tokens = tf.keras.utils.pad_sequences(tokens, maxlen=maxlen)
        
        if labels is not None:
            tokens = (tokens, labels)
        
        return tf.data.Dataset.from_tensor_slices(tokens)
    


    def load(self, path):
        with open(path) as f:
            json_string = json.load(f)
            self._tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json_string)

    def export(self, path, vocab=True):
        """
            Exports tokenizer config. If vocab, exports net vocabulary separately
        """
        path += "tokenizer"
        with open(path+".json", 'w') as f:
            json.dump(self._tokenizer.to_json(), f)
            print("... Tokenizer exported at %s" % path+".json")

        if vocab:
            self._export_vocab(path+"-vocab.json")
            print("... Vocabulary exported at %s" % path+"-vocab.json")

    def fit(self, rawdocs):  
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.max_terms)
        tokenizer.fit_on_texts(rawdocs)

        self._tokenizer = tokenizer

    def transform(self, rawdocs):
        #TODO
        pass

