from os.path import exists
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline
import scipy
import pickle
# pd.options.display.max_colwidth=300

df = pd.read_csv(
    "input/jigsaw-toxic-comment-classification-challenge/train.csv")
print(df.shape)
df_val = pd.read_csv("input/jigsaw-toxic-severity-rating/validation_data.csv")


# Give more weight to severe toxic
df['severe_toxic'] = df.severe_toxic * 2
df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat',
           'insult', 'identity_hate']].sum(axis=1)).astype(int)
df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
print(df.sample(5))
print(df['y'].value_counts())

# Reduce the rows with 0 toxicity
df = pd.concat([df[df.y > 0],
                df[df.y == 0].sample(int(len(df[df.y > 0])*1.5))], axis=0).sample(frac=1)
print(df.shape)
print(df['y'].value_counts())

# Create Sklearn Pipeline with TFIDF - Ridge
pipeline = Pipeline(
    [
        ("vect", TfidfVectorizer(min_df=3, max_df=0.5,
         analyzer='char_wb', ngram_range=(3, 5))),
        #("clf", RandomForestRegressor(n_estimators = 5, min_sample_leaf=3)),
        ("clf", Ridge()),
        # ("clf",LinearRegression())
    ]
)


if not exists('saved_model/tfidf_ridge.pickle'):
    # Train the pipeline
    pipeline.fit(df['text'], df['y'])

    # save model
    f = open('saved_model/tfidf_ridge.pickle', 'wb')
    pickle.dump(pipeline, f)
    f.close()

# load model
f = open('saved_model/tfidf_ridge.pickle', 'rb')
pipeline = pickle.load(f)
f.close()


# Validate the pipeline
p1 = pipeline.predict(df_val['less_toxic'])
p2 = pipeline.predict(df_val['more_toxic'])

# step 1 ACC
print(f'Validation Accuracy is { np.round((p1 < p2).mean() * 100,2)}')

df_val['p1'] = p1
df_val['p2'] = p2
df_val['diff'] = np.abs(p2 - p1)

df_val['correct'] = (p1 < p2).astype('int')