from tokenizer import Tokenizer
import pandas as pd
import tensorflow as tf
from abc import ABC, abstractmethod

class DocDataset(ABC):
    pass 


class DatasetFromCSV(DocDataset):
    def __init__(
            self,
            datapath,
            doccol,
            labelcol,
            splits,
            tokenizer=None,
            tok_maxterms=20000,
            tok_export=None,
            shuffle=True,
            useall=True):
        fulldf_ = pd.read_csv(datapath)[[doccol, labelcol]]

        if shuffle:
            fulldf_.sample(frac=1)

        _data_len = len(fulldf_)
        splits = list(map(lambda x : int(x*_data_len), splits))

        dfs = []
        start_split = 0
        for split in splits:
            end_split = start_split + split
            dfs.append(fulldf_.iloc[start_split:end_split])
            start_split += split

        # Additional split takes remaining data
        if useall and sum(splits) < _data_len:
            dfs.append(fulldf_.iloc[start_split:])

        self.fulldf_ = fulldf_
        self.doccol = doccol
        self.labelcol = labelcol
        self._data_len = _data_len
        self.dfs = dfs
        self._tf_datasets = None
        self.tokenizer = tokenizer
        self.tok_maxterms = tok_maxterms
        self.tok_export = tok_export

    def __len__(self):
        return self._data_len
    
    def generate_tf(self, batch_size=32, maxlen=None):
        tokenizer = self.tokenizer
        if tokenizer is None:
            tokenizer = Tokenizer(max_terms=self.tok_maxterms)
            tokenizer.fit(self.fulldf_[self.doccol])

            if self.tok_export:
                # Exports generated tokenizer and vocabulary
                tokenizer.export(self.tok_export)

        if self._tf_datasets is None:
            tf_datasets = []
            doccol = self.doccol
            labelcol = self.labelcol
            for df in self.dfs:
                tf_ds = tokenizer.texts_to_sequences(
                    texts=df[doccol].values,
                    labels=df[labelcol].values,
                    maxlen=maxlen
                )
                tf_ds = tf_ds\
                    .cache()\
                    .batch(batch_size)\
                    .prefetch(buffer_size=tf.data.AUTOTUNE)
                
                tf_datasets.append(tf_ds)

            self._tf_datasets = tf_datasets

    def export(self):
        # TODO: Export as pickle (only dataframes by default)
        pass

    @property
    def shape(self):
        return self._shape
    
    @shape.setter
    def shape(self, s):
        # TODO: Handle custom ds shape
        self._shape = s

    @property
    def tf_datasets(self):
        if not self._tf_datasets:
            raise ValueError("Run generate_tf to resolve the Tensorflow datasets")

        return self._tf_datasets
    
    @tf_datasets.setter
    def tf_datasets(self, tf_ds):
        self._tf_datasets = tf_ds




if __name__ == "__main__":
    dds = DatasetFromCSV("data/imdb.csv", "text", "label", (.7, .15), tok_export="exports/")
    print(len(dds))

    for df in dds.dfs:
        print(len(df))

    dds.generate_tf(batch_size=64, maxlen=100)