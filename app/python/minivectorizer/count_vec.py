import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from csr_matrix import MiniSparseMatrix

########### Dummy data
corpus = [
    "My cat is in the oven with her cat",
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

# Simplified count vectorize
class MiniCountVectorizer():
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        pass

    def _count_vocab():
        pass

    def fit_transform():
        pass

    def transform(self, tokenized_docs):
        feature_indices = []
        feature_counts = []
        indptr = [0]

        for tkd in tokenized_docs:
            feature_counter = {}

            for feature in tkd:
                # Tries to retrieve "feature" from vocabulary
                feature_index = self.vocabulary.get(feature, None)

                if feature_index:
                    # Feature in vocabulary. Updates counts
                    feature_count = feature_counter.get(feature_index, 0) + 1
                    feature_counter[feature_index] = feature_count

                else:
                    # Ignores out of vocabulary item
                    continue

            # Updates future csr matrix data, indices and indptr
            feature_indices.extend(feature_counter.keys())
            feature_counts.extend(feature_counter.values())
            indptr.append(len(feature_indices))

        # Transforms csr matrix data into np array form
        feature_indices = np.asarray(feature_indices, dtype=np.int32)
        feature_counts = np.asarray(feature_counts, dtype=np.int32)
        indptr = np.asarray(feature_counts, dtype=np.int32)
        X = MiniSparseMatrix(
            data=feature_counts,
            indices=feature_indices,
            indptr=indptr,
            shape=(len(indptr) - 1, len(self.vocabulary))
        )
        X.sort_indices()

        return X

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



ddata = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
dindices = np.array([2, 1, 5, 3, 9, 2, 1, 9, 4, 6, 0, 2, 3, 7, 8])
dindptr = np.array([0,  5,  9, 11, 14, 15])
dshape=(len(dindptr) - 1, 10) # vocabulary len = 10

msm = MiniSparseMatrix(data=ddata, indices=dindices, indptr=dindptr, shape=dshape)

print("----")
print(msm)

"""
Expected output:

  (0, 2)        1
  (0, 1)        2
  (0, 5)        1
  (0, 3)        1
  (0, 9)        1
  (1, 2)        1
  (1, 1)        1
  (1, 9)        1
  (1, 4)        1
  (2, 6)        1
  (2, 0)        1
  (3, 2)        1
  (3, 3)        1
  (3, 7)        1
  (4, 8)        1
[1 2 1 1 1 1 1 1 1 1 1 1 1 1 1]
[2 1 5 3 9 2 1 9 4 6 0 2 3 7 8]
[ 0  5  9 11 14 15]

"""