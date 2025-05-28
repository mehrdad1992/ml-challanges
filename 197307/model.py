import re
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import numpy as np


def get_avg_w2v(tokens, model, vector_size):
    valid_tokens = [token for token in tokens if token in model.wv]
    if not valid_tokens:
        return np.zeros(vector_size)
    return np.mean([model.wv[token] for token in valid_tokens], axis=0)


def main():
    df_train = pd.read_csv("dataset/train_data.csv")
    df_test = pd.read_csv("dataset/test_data.csv")

    delimiters = r'[;, \u200c&qout]' 
    df_train['title_tokens'] = [re.split(delimiters, df_train['title'].iloc[i]) for i, _ in enumerate(df_train['title'])]
    df_train['description_tokens'] = [re.split(delimiters, df_train['description'].iloc[i]) for i, _ in enumerate(df_train['description'])]
    df_test['title_tokens'] = [re.split(delimiters, df_test['title'].iloc[i]) for i, _ in enumerate(df_test['title'])]
    df_test['description_tokens'] = [re.split(delimiters, df_test['description'].iloc[i]) for i, _ in enumerate(df_test['description'])]

    all_tokens = pd.concat([df_train['title_tokens'], df_train['description_tokens']], ignore_index=True)
    w2v_model = Word2Vec(sentences=all_tokens, vector_size=100, window=5, min_count=1, workers=4)
    df_train['title_vec'] = df_train['title_tokens'].apply(lambda x: get_avg_w2v(x, w2v_model, 100))
    df_train['description_vec'] = df_train['description_tokens'].apply(lambda x: get_avg_w2v(x, w2v_model, 100))
    df_test['title_vec'] = df_test['title_tokens'].apply(lambda x: get_avg_w2v(x, w2v_model, 100))
    df_test['description_vec'] = df_test['description_tokens'].apply(lambda x: get_avg_w2v(x, w2v_model, 100))

    X_train = df_train.loc[:, ['title_vec', 'description_vec']].to_numpy()
    X_train = np.array([np.concatenate(row) for row in X_train])
    Y_train = df_train.loc[:, 'tags'].to_numpy()
    X_test = df_test.loc[:, ['title_vec', 'description_vec']].to_numpy()
    X_test = np.array([np.concatenate(row) for row in X_test])

    encoder = LabelEncoder()
    Y_encoded = encoder.fit_transform(Y_train)

    # Model Training
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_encoded)

    Y_pred = knn.predict(X_test)
    
    decoded_Y_preds = encoder.inverse_transform(Y_pred)

    # Save to CSV
    df = pd.DataFrame({'prediction': decoded_Y_preds})
    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()