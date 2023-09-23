#####
# Snippet used to train a model via sklearn. The resulting classifier will run on the front end
#####

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

# Glovars
datapath = "imdb.csv"
exportpath = "model_params.json"

def predict(sentences, model, vectorizer):
    # Handles sentence wrap
    sentences = sentences if isinstance(sentences, list) else [sentences]
    vectorized = vectorizer.transform(sentences)

    return model.predict(vectorized)

df = pd.read_csv(datapath)
df = df[:20]

# NOTE: datalabels are "review" & "sentiment" for the sentences and its labels respectively.

# Handle labels
df.sentiment.replace('positive', 1, inplace=True)
df.sentiment.replace('negative', 0, inplace=True)
df.sentiment = pd.to_numeric(df.sentiment, downcast='float')

# Implement a text preprocessing step before!!!
# TODO: Without this step, the model is full of doodoo

# Creates training sets
X_train, X_test, y_train, y_test = train_test_split(df.review, df.sentiment)


# Model training
print("training...")
vectorizer = CountVectorizer()
model = LogisticRegression(solver='lbfgs')
X_train_vect = vectorizer.fit_transform(X_train)

X_test_vect = vectorizer.transform(X_test)

print(type(X_test_vect))
print("shape of raw test set: %s, shape of sparse is %s" % (X_test.shape, X_test_vect.shape))
print("csr data shape: %s" % X_test_vect.data.shape)
print("csr indices shape: %s" % X_test_vect.indices.shape)
print("csr indptr shape: %s" % X_test_vect.indptr.shape) # indptr shape = nrows + 1 = ndocs + 1
exit()

"""
vectVocab = vectorizer.vocabulary_ # this is a dict
vocabRandKey = list(vectVocab.keys())[42]
vocabThingKey = "thing"

print("n keys: %d" % len(list(vectVocab.keys())))
print("vocabulary random key (%s) value: %s" % (vocabRandKey, vectVocab[vocabRandKey]))

exit()
"""
model.fit(X_train_vect, y_train)

# Test
print("testing...")
#This halts the code. Do not fit again!! X_test_vect = vectorizer.fit_transform(X_test)
X_test_vect = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vect)
print("F1 score: ", f1_score(y_test, y_pred))

print(predict("this movie is great!", model, vectorizer))
print(predict("this movie is awful", model, vectorizer))
print(predict("worst movie ever", model, vectorizer))

# Export model parameters
import json
feature_names = vectorizer.get_feature_names()

d = dict()
d['words'] = feature_names
d['values'] = model.coef_.tolist()
d['intercept'] = model.intercept_.tolist()
d['classes'] = model.classes_.tolist()

with open(exportpath, "w") as f:
    f.write(json.dumps(d))


"""
print(df['sentiment'].unique())
print(df.sentiment.value_counts())
"""

print("done! :D")