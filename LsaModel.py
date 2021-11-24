from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import collections
import numpy as np
from collections import Counter


class LsaModel:

    def __init__(self, text_tokenize):
        self.docs = text_tokenize
        self.lsa_matrix = [[]]

    def calc_doc_term_matrix(self):
        intptr = [0]
        indices = []
        data = []
        vocabulary = {}
        for d in self.docs:
            for term in d:
                index = vocabulary.setdefault(term, len(vocabulary))
                indices.append(index)
                data.append(1)
            intptr.append(len(indices))

        return csr_matrix((data, indices, intptr), dtype=int)

    def get_tf_idf_matrix(self, dtm_matrix):
        number_of_all_documents = len(self.text_tokenize)
        row_indices, col_indices = dtm_matrix.nonzero()
        term_occurences = np.array(list(Counter(col_indices).values()))
        number_of_words_in_text = dtm_matrix.sum(axis=1)
        number_of_words_in_text_transpose = number_of_words_in_text.transpose()
        number_of_words_in_text_array = np.squeeze(np.asarray(number_of_words_in_text_transpose))
        matrix_inverse_number_of_words = csr_matrix(
            ((1.0 / number_of_words_in_text_array)[row_indices], (row_indices, col_indices)), shape=(dtm_matrix.shape))
        tf_matrix = dtm_matrix.multiply(matrix_inverse_number_of_words)
        idf = np.asarray([np.log(number_of_all_documents / item) for item in term_occurences])
        row_indices, col_indices = tf_matrix.nonzero()

        idf_matrix = csr_matrix(((np.asarray(idf))[col_indices], (row_indices, col_indices)), shape=(dtm_matrix.shape))
        return tf_matrix.multiply(idf_matrix)

    def get_lsa_matrix(self):
        dtm_matrix = self.calc_doc_term_matrix()
        tf_idf_matrix = self.get_tf_idf_matrix(dtm_matrix)
        svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
        self.lsa_matrix = svd.fit_transform(tf_idf_matrix)
        return self.lsa_matrix

    #through cosinus calculate similarity
    def calc_map_metric(self, k, targett):
        row_sums = self.lsa_matrix.sum(axis=1)
        norm_dtm_svd_matrix = self.lsa_matrix / row_sums[:, np.newaxis]
        dtm_svd_matrix_transpose = self.lsa_matrix.transpose()
        dot_norm_transpose = norm_dtm_svd_matrix.dot(dtm_svd_matrix_transpose)
        np.fill_diagonal(dot_norm_transpose, -1)
        indexes_of_docs = np.argpartition(-dot_norm_transpose, axis=0, kth=k)
        indexes_of_most_similar_docs = indexes_of_docs[:indexes_of_docs.shape[0], :k]
        compare_group = np.add(targett[indexes_of_most_similar_docs].T, -targett)
        return np.mean(np.count_nonzero(compare_group == 0, axis=0) / k)