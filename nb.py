import numpy as np
import time
from tqdm import tqdm

class SpamFilter():
    def __init__(self):
        print("SpamFilter Object")

    def train(self, emails):
        # assuming that parameter emails contains both spam and ham
        # result = {} # store word and frequency
        spam_freq = {}
        ham_freq = {}

        # train_count = len(emails)
        count_spam = 0
        count_ham = 0
        # count_email = 0

        # iterate through emails
        for email in tqdm(emails):

            # deleted duplicates from a single email before start counting
            text = set(email[0])
            label = email[-1]

        # create dict spam_freq, ham_freq
            for token in text:
                if label == "spam":
                    count_spam += 1
                    if token in spam_freq.keys():
                        spam_freq[token] += 1
                    else:
                        spam_freq[token] = 1

                if label == "ham":
                    count_ham += 1
                    if token in ham_freq.keys():
                        ham_freq[token] += 1
                    else:
                        ham_freq[token] = 1

        # create a list of vocab with no duplicate
        self.vocab = list(dict.fromkeys(list(ham_freq.keys()) + list(spam_freq.keys())))

        # calculate equation (4)
        self.p_word_spam = np.zeros(len(self.vocab)) # probabilities of a word appear in spam email
        self.p_word_ham = np.zeros(len(self.vocab)) # probabilities of a word appear in spam email

        for index, word in enumerate(self.vocab):
            # calculate probability of a word appear in spam email
            prob = spam_freq.get(word, 0) / count_spam
            self.p_word_spam[index] = prob
            # calculate probability of a word appear in ham email
            prob = ham_freq.get(word, 0) / count_ham
            self.p_word_ham[index] = prob

    def classify(self, email):
        p_spam = []
        p_ham = []

        # read email
        for token in email:
            # check if word exist in vocab
            if token in self.vocab:
                index = self.vocab.index(token)
                # +1 to avoid multiplying by zero and floating point underflow
                p_spam.append(self.p_word_spam[index] + 1)
                p_ham.append(self.p_word_ham[index] + 1)
            else:
                # skip word if not found in training data
                pass

        # calculate z
        z = (np.prod(p_spam)) + (np.prod(p_ham))

        # calculate spam score
        spam_score = (1 / z) * np.prod(p_spam)

        # define threshold and classify
        threshold = 0.49
        if spam_score >= threshold:
            result = "spam"
        else:
            result = "ham"

        return result, spam_score
