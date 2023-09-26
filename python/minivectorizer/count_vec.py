# Simplified CountVectorizer from sklearn
# See https://github.com/scikit-learn/scikit-learn/blob/55a65a2fa/sklearn/feature_extraction/text.py#L1343

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from csr_matrix import DocSparseMatrix
from numbers import Integral

########### Dummy data
corpus = [
    "My cat is in my oven with my cat",
    "I love taking long strolls with my cat",
    "They both get along just like dogs and cats",
    "I think I left my oven on",
    "How many hours to a leap year?"
]

corpus_test = [
    "I have not seen Julie's cat in a year",
    "Do you recon I can fix the oven?"
]
max_features = 10
######################

class MiniCountVectorizer():
    def __init__(
        self,
        vocabulary=None,
        max_df=1.0,
        min_df=1,
        max_features=None
    ):
        self.vocabulary_ = vocabulary
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features

    def _count_vocab(self, tokenized_docs, fixed_vocab=True):
        """
            Creates sparse matrix out of tokenized docs. Also populates vocabulary if !fixed_vocab.

            PARAMS:
            tokenized_docs: Iterable over tokenized text documents.
                Each sample must be a collection of tokens.

            RETURN:
            vocabulary: A dictionary representing the active vocabulary
            X: sparse matrix from tokenized_docs
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_

        else:
            vocabulary = {}
            # vocabulary will update with unseen keys rather than raising an error
            # Default key value is the vocabulary lenght, which will be the new token id
            #vocabulary = defaultdict()
            #vocabulary.default_factory = vocabulary.__len__

        feature_indices = []
        feature_counts = []
        indptr = [0]

        for tkd in tokenized_docs:
            feature_counter = {}

            for feature in tkd:
                # Tries to retrieve "feature" from vocabulary
                feature_index = vocabulary.get(feature, None)

                if feature_index:
                    # Feature in vocabulary. Updates counts
                    feature_count = feature_counter.get(feature_index, 0) + 1
                    feature_counter[feature_index] = feature_count

                else:
                    # Out of vocabulary item
                    if fixed_vocab:
                        continue

                    #Default key value is the vocabulary lenght, which will be the new token id
                    vocabulary[feature] = len(vocabulary)
                    
            # Updates future csr matrix data, indices and indptr
            feature_indices.extend(feature_counter.keys())
            feature_counts.extend(feature_counter.values())
            indptr.append(len(feature_indices))

        # Transforms csr matrix data into np array form
        feature_indices = np.asarray(feature_indices, dtype=np.int32)
        feature_counts = np.asarray(feature_counts, dtype=np.int32)
        indptr = np.asarray(feature_counts, dtype=np.int32)
        X = DocSparseMatrix(
            data=feature_counts,
            indices=feature_indices,
            indptr=indptr
            #shape=(len(indptr) - 1, len(vocabulary))
        )
        
        X.sort_indices()
        return vocabulary, X
    
    def _sort_features(self, X, vocabulary):
        """
            Sorts vocabulary features by name in place. Updates indices in document matrix

            ----------------
            vocab:   c d b a
            indices: 2 1 0 3

            -- sort --

            vocab:   c d b a
            indices: 2 3 1 0
            ----------------

            PARAMS:
            x: Document sparse matrix
            vocabulary: A dictionary representing the active vocabulary

            RETURN:
            x: Updated matrix
        """
        sorted_features = sorted(vocabulary.items())
        map_index = np.empty(len(sorted_features), dtype=X.indices.dtype)
        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            map_index[old_val] = new_val

        X.indices = map_index.take(X.indices, mode="clip")
        return X

    def _limit_features(self, x, vocabulary, high=None, low=None, limit=None):
        # TODO: Replicating https://github.com/scikit-learn/scikit-learn/blob/55a65a2fa5653257225d7e184da3d0c00ff852b1/sklearn/feature_extraction/text.py#L1204
        """
            Removes too rare or too common features.
            Also restricts vocabulary to at most the "limit" most frequent features
        """
        dfq = x._document_frequency() # This returns an array representing the term document frequency
        mask = np.ones(len(dfq), dtype=bool)

        # Bool mask from specified DOCUMENT frequency tresholds
        if high is not None:
            mask &= dfq <= high
        if low is not None:
            mask &= dfq >= low
        
        if limit is not None and mask.sum() > limit:
            # TODO: This should generate another mask from the most frequent TERMS across the corpus
            # Compute term frequency
            pass

        # TODO: return pruned vocabulary and csr matrix. Also return removed terms

    def fit_transform(self, tokenized_docs):
        """
            Learns vocabulary and returns document-term matrix.

            PARAMS:
            tokenized_docs: Iterable over tokenized text documents.
                Each sample must be a collection of tokens.

            RETURN:
            x: sparse matrix from tokenized_docs of shape (n_docs, n_features)
        """
        max_df = self.max_df
        min_df = self.min_df
        max_features = self.max_features

        vocabulary, X = self._count_vocab(tokenized_docs, fixed_vocab=False)

        n_doc = X.shape[0]
        max_doc_count = max_df if isinstance(max_df, Integral) else max_df * n_doc
        min_doc_count = min_df if isinstance(min_df, Integral) else min_df * n_doc

        if max_doc_count < min_doc_count:
            raise ValueError("max_df corresponds to less documents than min_df")

        # TODO: sort features, limit features 

        self.vocabulary_ = vocabulary
        return X

    def transform(self, tokenized_docs):
        _, x = self._count_vocab(tokenized_docs)

        return x

"""
NOTE:
Every option related to text processing must be part of a standalone preprocessor
(strip_accents, lowercase, tokenizer, stop_words, token_pattern, ngram_range)

The vectorizer simply takes the already tokenized input and turns it into a vector using options
(max_df, min_df, max_features)

token_pattern='(?u)\b\w\w+\b',  # Regex to denote what constitutes a word (if analyzer=word)
analyzer='word', # Whether to generate features out of word n-grams or char n-grams

"""
vectorizer = CountVectorizer(max_features=max_features)

X_vec = vectorizer.fit_transform(corpus)
print(vectorizer.vocabulary_)
print("-----------------")
print(X_vec)
print(X_vec.data)
print(X_vec.indices)
print(X_vec.indptr)

print("-----------------")

# Gives wrong info
for crsm in X_vec:
    print(crsm)
    print(crsm.data)
    print(crsm.indices)
    print(crsm.indptr)
    print("...")



ddata = np.array([3, 2, 1, 1,   1, 1, 1, 1, 1,   1, 1,   1, 1, 1,   1])
dindices = np.array([2, 1, 3, 9,   2, 1, 9, 5, 4,   6, 0,   2, 3, 7,   8])
dindptr = np.array([0,  4,  9, 11, 14, 15])
dshape=(len(dindptr) - 1, 10) # vocabulary len = 10

msm = DocSparseMatrix(data=ddata, indices=dindices, indptr=dindptr)

print("----")
print(msm)

print(msm.shape)

print(vectorizer.vocabulary_)
print(sorted(vectorizer.vocabulary_))
print(vectorizer._sort_features(msm, vectorizer.vocabulary_).indices)
print(vectorizer.vocabulary_)

print("--- sorted csr")
msm.sort_indices()
print(msm)

print(vectorizer.vocabulary_)
print(sorted(vectorizer.vocabulary_))

print(np.bincount(msm.indices))

"""
Expected output:

  (0, 2)        3
  (0, 1)        2
  (0, 3)        1
  (0, 9)        1
  (1, 2)        1
  (1, 1)        1
  (1, 9)        1
  (1, 5)        1
  (1, 4)        1
  (2, 6)        1
  (2, 0)        1
  (3, 2)        1
  (3, 3)        1
  (3, 7)        1
  (4, 8)        1
[3 2 1 1 1 1 1 1 1 1 1 1 1 1 1]
[2 1 3 9 2 1 9 5 4 6 0 2 3 7 8]
[ 0  4  9 11 14 15]

--- sorted csr

1 2 3 9 . 1 2 4 5 9 . 0 6 . 2 3 7 . 8
2 3 1 1 . 1 1 1 1 1 . 1 1 . 1 1 1 . 1

"""