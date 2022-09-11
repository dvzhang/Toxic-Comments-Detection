import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
import re
import scipy
from scipy import sparse
import gc
from IPython.display import display, HTML
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_colwidth = 300


# NOT USED
class LengthTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return sparse.csr_matrix([[(len(x)-360)/550] for x in X])

    def get_feature_names(self):
        return ["lngth"]


class LengthUpperTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return sparse.csr_matrix([[sum([1 for y in x if y.isupper()])/len(x)] for x in X])

    def get_feature_names(self):
        return ["lngth_uppercase"]


def clean(data, col):

    # Clean some punctutations
    data[col] = data[col].str.replace('\n', ' \n ')
    data[col] = data[col].str.replace(
        r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)', r'\1 \2 \3')
    # Replace repeating characters more than 3 times to length of 3
    data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}', r'\1\1\1')
    # Add space around repeating characters
    data[col] = data[col].str.replace(r'([*!?\']+)', r' \1 ')
    # patterns with repeating characters
    data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b', r'\1\1')
    data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B', r'\1\1\1')
    data[col] = data[col].str.replace(r'[ ]{2,}', ' ').str.strip()

    return data


df = pd.read_csv(
    "input/jigsaw-toxic-comment-classification-challenge/train.csv")
print(df.shape)

for col in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    print(f'****** {col} *******')
    display(df.loc[df[col] == 1, ['comment_text', col]].sample(10))

# Give more weight to severe toxic
df['severe_toxic'] = df.severe_toxic * 2
df['y'] = (df[['toxic', 'severe_toxic', 'obscene', 'threat',
           'insult', 'identity_hate']].sum(axis=1)).astype(int)
df['y'] = df['y']/df['y'].max()

df = df[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
print(df.sample(5))
print(df['y'].value_counts())

# Create 3 versions of the data
n_folds = 5
frac_1 = 0.8
frac_1_factor = 1.5
for fld in range(n_folds):
    print(f'Fold: {fld}')
    tmp_df = pd.concat([df[df.y > 0].sample(frac=frac_1, random_state=10*(fld+1)),
                        df[df.y == 0].sample(n=int(len(df[df.y > 0])*frac_1*frac_1_factor),
                                             random_state=10*(fld+1))], axis=0).sample(frac=1, random_state=10*(fld+1))

    tmp_df.to_csv(f'tmpData/tfidf_ridge_2data/df_fld{fld}.csv', index=False)
    print(tmp_df.shape)
    print(tmp_df['y'].value_counts())


# Create 3 versions of clean data
test_clean_df = pd.DataFrame({"text":
                              ["heyy\n\nkkdsfj",
                               "hi   how/are/you ???",
                               "hey?????",
                               "noooo!!!!!!!!!   comeone !! ",
                               "cooooooooool     brooooooooooo  coool brooo",
                               "naaaahhhhhhh"]})
display(test_clean_df)
clean(test_clean_df, 'text')

df = clean(df, 'text')
n_folds = 5
frac_1 = 0.8
frac_1_factor = 1.5
for fld in range(n_folds):
    print(f'Fold: {fld}')
    tmp_df = pd.concat([df[df.y > 0].sample(frac=frac_1, random_state=10*(fld+1)),
                        df[df.y == 0].sample(n=int(len(df[df.y > 0])*frac_1*frac_1_factor),
                                             random_state=10*(fld+1))], axis=0).sample(frac=1, random_state=10*(fld+1))

    tmp_df.to_csv(
        f'tmpData/tfidf_ridge_2data/df_clean_fld{fld}.csv', index=False)
    print(tmp_df.shape)
    print(tmp_df['y'].value_counts())

# read ruddit data
df_ = pd.read_csv("input/ruddit_jigsaw_dataset/Dataset/ruddit_with_text.csv")
print(df_.shape)
df_ = df_[['txt', 'offensiveness_score']].rename(columns={'txt': 'text',
                                                          'offensiveness_score': 'y'})

df_['y'] = (df_['y'] - df_.y.min()) / (df_.y.max() - df_.y.min())
df_.y.hist()
n_folds = 5

frac_1 = 0.8

for fld in range(n_folds):
    print(f'Fold: {fld}')
    tmp_df = df_.sample(frac=frac_1, random_state=10*(fld+1))
    tmp_df.to_csv(f'tmpData/tfidf_ridge_2data/df2_fld{fld}.csv', index=False)
    print(tmp_df.shape)
    print(tmp_df['y'].value_counts())
del tmp_df, df_
gc.collect()

# Validation data
df_val = pd.read_csv(
    "input/jigsaw-toxic-severity-rating/validation_data.csv")
# Test data
df_sub = pd.read_csv(
    "input/jigsaw-toxic-severity-rating/comments_to_score.csv")

df_val['upper_1'] = np.array(LengthUpperTransformer().transform(
    df_val['less_toxic']).todense()).reshape(-1, 1)
df_val['upper_2'] = np.array(LengthUpperTransformer().transform(
    df_val['more_toxic']).todense()).reshape(-1, 1)

print(df_val['upper_1'].mean(), df_val['upper_1'].std())
print(df_val['upper_2'].mean(), df_val['upper_2'].std())

df_val['upper_1'].hist(bins=100)
df_val['upper_2'].hist(bins=100)

val_preds_arr1 = np.zeros((df_val.shape[0], n_folds))
val_preds_arr2 = np.zeros((df_val.shape[0], n_folds))
test_preds_arr = np.zeros((df_sub.shape[0], n_folds))

for fld in range(n_folds):
    print("\n\n")
    print(
        f' ****************************** FOLD: {fld} ******************************')
    df = pd.read_csv(f'tmpData/tfidf_ridge_2data/df_fld{fld}.csv')
    print(df.shape)

    features = FeatureUnion([
        #('vect1', LengthTransformer()),
        #('vect2', LengthUpperTransformer()),
        ("vect3", TfidfVectorizer(min_df=3, max_df=0.5,
         analyzer='char_wb', ngram_range=(3, 5))),
        #("vect4", TfidfVectorizer(min_df= 5, max_df=0.5, analyzer = 'word', token_pattern=r'(?u)\b\w{8,}\b')),

    ])
    pipeline = Pipeline(
        [
            ("features", features),
            #("clf", RandomForestRegressor(n_estimators = 5, min_sample_leaf=3)),
            ("clf", Ridge()),
            # ("clf",LinearRegression())
        ]
    )
    print("\nTrain:")
    # Train the pipeline
    pipeline.fit(df['text'], df['y'])

    # What are the important features for toxicity

    print('\nTotal number of features:', len(
        pipeline['features'].get_feature_names()))

    feature_wts = sorted(list(zip(pipeline['features'].get_feature_names(),
                                  np.round(pipeline['clf'].coef_, 2))),
                         key=lambda x: x[1],
                         reverse=True)

    pprint(feature_wts[:30])

    print("\npredict validation data ")
    val_preds_arr1[:, fld] = pipeline.predict(df_val['less_toxic'])
    val_preds_arr2[:, fld] = pipeline.predict(df_val['more_toxic'])

    print("\npredict test data ")
    test_preds_arr[:, fld] = pipeline.predict(df_sub['text'])
val_preds_arr1c = np.zeros((df_val.shape[0], n_folds))
val_preds_arr2c = np.zeros((df_val.shape[0], n_folds))
test_preds_arrc = np.zeros((df_sub.shape[0], n_folds))

for fld in range(n_folds):
    print("\n\n")
    print(f' ****************************** FOLD: {fld} ******************************')
    df = pd.read_csv(f'tmpData/tfidf_ridge_2data/df_clean_fld{fld}.csv')
    print(df.shape)

    features = FeatureUnion([
        #('vect1', LengthTransformer()),
        #('vect2', LengthUpperTransformer()),
        ("vect3", TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5))),
        #("vect4", TfidfVectorizer(min_df= 5, max_df=0.5, analyzer = 'word', token_pattern=r'(?u)\b\w{8,}\b')),

    ])
    pipeline = Pipeline(
        [
            ("features", features),
            #("clf", RandomForestRegressor(n_estimators = 5, min_sample_leaf=3)),
            ("clf", Ridge()),
            #("clf",LinearRegression())
        ]
    )
    print("\nTrain:")
    # Train the pipeline
    pipeline.fit(df['text'], df['y'])
    
    # What are the important features for toxicity

    print('\nTotal number of features:', len(pipeline['features'].get_feature_names()) )

    feature_wts = sorted(list(zip(pipeline['features'].get_feature_names(), 
                                  np.round(pipeline['clf'].coef_,2) )), 
                         key = lambda x:x[1], 
                         reverse=True)

    pprint(feature_wts[:30])
    
    print("\npredict validation data ")
    val_preds_arr1c[:,fld] = pipeline.predict(df_val['less_toxic'])
    val_preds_arr2c[:,fld] = pipeline.predict(df_val['more_toxic'])

    print("\npredict test data ")
    test_preds_arrc[:,fld] = pipeline.predict(df_sub['text'])

val_preds_arr1_ = np.zeros((df_val.shape[0], n_folds))
val_preds_arr2_ = np.zeros((df_val.shape[0], n_folds))
test_preds_arr_ = np.zeros((df_sub.shape[0], n_folds))

for fld in range(n_folds):
    print("\n\n")
    print(f' ****************************** FOLD: {fld} ******************************')
    df = pd.read_csv(f'/kaggle/working/df2_fld{fld}.csv')
    print(df.shape)

    features = FeatureUnion([
        #('vect1', LengthTransformer()),
        #('vect2', LengthUpperTransformer()),
        ("vect3", TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5))),
        #("vect4", TfidfVectorizer(min_df= 5, max_df=0.5, analyzer = 'word', token_pattern=r'(?u)\b\w{8,}\b')),

    ])
    pipeline = Pipeline(
        [
            ("features", features),
            #("clf", RandomForestRegressor(n_estimators = 5, min_sample_leaf=3)),
            ("clf", Ridge()),
            #("clf",LinearRegression())
        ]
    )
    print("\nTrain:")
    # Train the pipeline
    pipeline.fit(df['text'], df['y'])
    # What are the important features for toxicity

    print('\nTotal number of features:', len(pipeline['features'].get_feature_names()) )

    feature_wts = sorted(list(zip(pipeline['features'].get_feature_names(), 
                                  np.round(pipeline['clf'].coef_,2) )), 
                         key = lambda x:x[1], 
                         reverse=True)

    pprint(feature_wts[:30])
    
    print("\npredict validation data ")
    val_preds_arr1_[:,fld] = pipeline.predict(df_val['less_toxic'])
    val_preds_arr2_[:,fld] = pipeline.predict(df_val['more_toxic'])

    print("\npredict test data ")
    test_preds_arr_[:,fld] = pipeline.predict(df_sub['text'])