import parsivar
from hazm import stopwords_list
import pandas as pd
import numpy as np
from parsivar import Normalizer
from parsivar import Tokenizer
from parsivar import FindStems
from timeit import default_timer
from sklearn.model_selection import train_test_split

VOCAB_SIZE = 3000


def read_data():
    data = pd.read_csv("digikala_comment/train_comments.csv", usecols=[2, 3, 4])
    return data


def clean_data(data_frame):
    null_index = (data_frame[data_frame.comment.isnull() == True].index.tolist())
    data_frame.drop(null_index, inplace=True)

    return data_frame


def is_alpha(word):
    alphabet = ['ا', 'آ', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ',
                'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و', 'ه', 'ی']

    for letter in word:
        if letter not in alphabet:
            return False
    return True


def process_comment(comment):
    normalizer = Normalizer(statistical_space_correction=True)
    tokenizer = Tokenizer()
    stemmer = FindStems()
    stop_words = set(stopwords_list())
    clean_token = []

    tokens = tokenizer.tokenize_words(normalizer.normalize(comment))
    for token in tokens:
        if token not in stop_words and is_alpha(token):
            stem_word = stemmer.convert_to_stem(token)
            if '&' in stem_word:
                clean_token.append(stem_word.split('&')[0])

            else:
                clean_token.append(stem_word)

    return clean_token


def find_frequent_tokens(tokens_list):
    frequent_tokens = pd.Series(tokens_list).value_counts()[:VOCAB_SIZE]
    # print(frequent_tokens.head())
    word_ids = list(range(VOCAB_SIZE))
    vocab = pd.DataFrame({'VOCAB_WORD': frequent_tokens.index.values}, index=word_ids)
    vocab.index.name = 'word_id'

    return vocab


def create_occurrence_matrix(df, word_index, labels):
    row = df.shape[0]
    col = df.shape[1]
    word_set = set(word_index)
    occurrence_list = []

    for i in range(row):
        for j in range(col):
            word = df.iat[i, j]
            if word in word_set:
                comment_id = df.index[i]
                word_id = word_index.get_loc(word)
                category = labels.at[comment_id]
                item = {'LABEL': category, 'COMMENT_ID': comment_id, 'WORD_ID': word_id, 'OCCURRENCE': 1}
                occurrence_list.append(item)

    return pd.DataFrame(occurrence_list)


if __name__ == "__main__":
    data = read_data()
    data = clean_data(data)
    label = data.verification_status.values
    label = pd.Series(label)
    spam_index = label[label == 0].index
    ham_index = label[label == 1].index

    filter_data = data.comment.apply(process_comment)

    df = pd.DataFrame.from_records(filter_data.tolist())
    df.to_csv('digikala_comments.csv', index=False)

    flat_data = [item for comment in filter_data for item in comment]
    frequent_words = find_frequent_tokens(flat_data)
    print(frequent_words.head())

    spam_comment = filter_data.loc[spam_index]
    ham_comment = filter_data.loc[ham_index]

    spam_flat_data = [item for subcomment in spam_comment for item in subcomment]
    ham_flat_data = [item for subcomment in ham_comment for item in subcomment]
    flat_data = [item for subcomment in filter_data for item in subcomment]

    flat_data = [item for index in range(data.shape[0]) for item in data.loc[index]]
    print(pd.Series(flat_data).value_counts().shape)
    print(pd.Series(flat_data).value_counts().size)
    frequent_words = pd.Series(flat_data).value_counts()[:VOCAB_SIZE]
    word_ids = list(range(VOCAB_SIZE))
    vocab_df = pd.DataFrame({'VOCAB_WORD': frequent_words.index.values}, index=word_ids)
    vocab_df.to_csv('digikala_frequent_words.csv', index_label='word_id', header='vocab_word')

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

    frequent_words = pd.read_csv('digikala_frequent_words.csv')
    vocabs = pd.Index(frequent_words.VOCAB_WORD)

    train_matrix = create_occurrence_matrix(X_train, vocabs, y_train)
    train_matrix = train_matrix.groupby(['COMMENT_ID', 'WORD_ID', 'LABEL']).sum()
    train_matrix = train_matrix.reset_index()
    np.savetxt("digikala_train_occurrence_matrix.txt", train_matrix, fmt='%d')

    test_matrix = create_occurrence_matrix(X_test, vocabs, y_test)
    test_matrix = test_matrix.groupby(['COMMENT_ID', 'WORD_ID', 'LABEL']).sum()
    test_matrix = test_matrix.reset_index()
    np.savetxt("digikala_test_occurrence_matrix.txt", test_matrix, fmt='%d')
