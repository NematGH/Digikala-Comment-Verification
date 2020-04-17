import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

VOCAB_SIZE = 3000

train_file = 'digikala_train_occurrence_matrix.txt'
test_file = 'digikala_test_occurrence_matrix.txt'

train_occurrence_matrix = np.loadtxt(train_file)
test_occurrence_matrix = np.loadtxt(test_file)


def create_full_sparse_matrix(occurrence_matrix, comment_id=0, word_id=1, label_id=2, frequent_number=3):
    col_name = ['COMMENT_ID'] + ['CATEGORY'] + list(range(VOCAB_SIZE))
    comment_id_names = np.unique(occurrence_matrix[:, comment_id])

    sparse_matrix = pd.DataFrame(index=comment_id_names, columns=col_name)
    sparse_matrix.COMMENT_ID = comment_id_names
    for i in range(sparse_matrix.shape[0]):
        doc = occurrence_matrix[i, comment_id]
        word = occurrence_matrix[i, word_id]
        category = occurrence_matrix[i, label_id]
        freq = occurrence_matrix[i, frequent_number]

        sparse_matrix.at[doc, 'CATEGORY'] = category
        sparse_matrix.at[doc, word] = freq

    sparse_matrix.set_index('DOC_ID', inplace=True)

    return sparse_matrix


matrix = create_full_sparse_matrix(train_occurrence_matrix)

y = matrix.CATEGORY
matrix.dropna(['CATEGORY'], axis=1, inplace=True)

naive = GaussianNB()
naive.fit(matrix, y)

test_full_matrix = create_full_sparse_matrix(test_occurrence_matrix)
y_t = test_full_matrix.CATEGORY
test_full_matrix.drop('CATEGORY', axis=1, inplace=True)
print(naive.score(test_full_matrix, y_t))
