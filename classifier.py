import numpy as np

"""
A supervised learning based classifier to determine whether a given email is spam or ham
"""


class SpamClassifier:
    def __init__(self, data, alpha=1.0):
        """
        Instantiates a Naïve Bayes classifier that determines whether a given email is spam or ham (not spam).

        :param data: the training data which is a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]
                     the first column contains the binary response (coded as 0s and 1s).

        :param alpha: Laplace-smoothing factor
        """
        self.training_data = data
        self.alpha = alpha
        self.log_class_priors = None
        self.log_class_conditional_likelihoods = None

    def train(self):
        """

        :return:
        """
        self.estimate_log_class_priors()
        self.estimate_log_class_conditional_likelihoods()

    def predict(self, data):
        """
        Predicts the corresponding response for each instance (row) of the data set.

        :param data: a two-dimensional numpy-array with shape = [n_test_samples, n_features].
        :return class_predictions: a numpy array containing the class predictions for each row of the data.
        """
        class_predictions = []

        # for each email
        for row in range(len(data)):
            # get columns/keywords that are present (i.e. 1) for this row
            cols_with_keywords = np.where(data[row] == 1)[0]

            # get the product of log of conditional likelihoods and log priors for each class
            ham_conditional_likelihoods_product = np.product(
                [self.log_class_conditional_likelihoods[0][i] for i in cols_with_keywords])
            ham_numerator = self.log_class_priors[0] * ham_conditional_likelihoods_product

            spam_conditional_likelihoods_product = np.product(
                [self.log_class_conditional_likelihoods[1][i] for i in cols_with_keywords])
            spam_numerator = self.log_class_priors[1] * spam_conditional_likelihoods_product

            # derive probability for each class
            denominator = spam_numerator + ham_numerator  # since (spam_numerator+ham_numerator)/denominator = 1
            spam_probability, ham_probability = spam_numerator / denominator, ham_numerator / denominator

            # classify the email based on the greater probability
            class_predictions.append(0 if ham_probability > spam_probability else 1)

        return np.array(class_predictions)

    def estimate_log_class_priors(self):
        """
        Calculates the logarithm of the empirical class priors, i.e. the logarithm of the proportions of 0s and 1s:
        log(p(C=0)) and log(p(C=1))

        Assigns the result to self.log_class_priors
        """
        # get first column of array which contains the binary response variables/labels
        np_data_labels = self.training_data[:, 0]

        # get total number of zeros and ones
        total = len(np_data_labels)
        n_zeros = np.count_nonzero(np_data_labels == 0)
        n_ones = total - n_zeros

        # calculate logarithm of class priors
        ham_priors, spam_priors = n_zeros / total, n_ones / total

        self.log_class_priors = np.array([np.log(ham_priors), np.log(spam_priors)])

    def estimate_log_class_conditional_likelihoods(self):
        """
        Calculates the empirical class-conditional likelihoods i.e. log(P(w_i | c)) for all features w_i and both classes (c in {0, 1}).

        Assumes a multinomial feature distribution and uses Laplace smoothing of [self.alpha]

        Assigns the result to [self.log_class_conditional_likelihoods]
            a numpy array of shape = [2, n_features]. self.log_class_conditional_likelihoods[j, i] corresponds to the
            logarithm of the probability of feature i appearing in a sample belonging
            to class j.
        """

        total_spam_word_count = total_ham_word_count = 0
        row_len, col_len = len(self.training_data), len(self.training_data[0])
        spam_result, ham_result = [], []

        # for each keyword, calculate how many times it appears in a class (spam or ham emails)
        # OR say for a class, calculate the count of the class' emails that contains the keyword.
        for col in range(1, col_len):
            spam_keyword_count, ham_keyword_count = 0.0, 0.0

            for row in range(row_len):
                if self.training_data[row][col] == 1:
                    #  if spam email
                    if self.training_data[row][0] == 1:
                        spam_keyword_count += 1
                        total_spam_word_count += 1
                    else:
                        ham_keyword_count += 1
                        total_ham_word_count += 1

            ham_result.append(ham_keyword_count)
            spam_result.append(spam_keyword_count)

        # perform laplace smoothing: (n(c,w) + alpha) / (n(w) + k*alpha) where k = no of keywords
        keywords = self.training_data.shape[1] - 1

        spam_result = [np.log((count + self.alpha) / (total_spam_word_count + (keywords * self.alpha))) for count in
                       spam_result]

        ham_result = [np.log((count + self.alpha) / (total_ham_word_count + (keywords * self.alpha))) for count in
                      ham_result]

        self.log_class_conditional_likelihoods = np.array([ham_result, spam_result])


def create_classifier():
    classifier = SpamClassifier(training_spam)
    classifier.train()
    return classifier


classifier = create_classifier()
