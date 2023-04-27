import os
import re
import nltk
from nltk.corpus import stopwords, words, names
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from tqdm import tqdm


def tokenize(text, remove_punctuation=True, remove_stopwords=True, check_nonwords=False):
    tokens = []

    # check if content of text is empty
    if text == "":
        return ["__NO_CONTENT__"]

    # if remove_punctuation is True
    # proceed to remove punctuation before tokenize
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)

    # tokenize
    tokenized_text = word_tokenize(text)

    # if remove_stopwords is True
    # proceed to remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        for token in tokenized_text:
            # remove stopwords
            if token not in stop_words:
                tokens.append(token)
    else:
        tokens = tokenized_text

    # if check_nonwords is True
    # proceed to check if token exists in dictionary
    if check_nonwords:
        # POS tagging
        # tagged consists of tuples ("word", "POS")
        tagged = nltk.pos_tag(tokens)

        # lemmatize
        lemmatizer = WordNetLemmatizer()
        mapped_word = []
        for each in tagged:
            # word = each[0], pos = each[1]
            if each[1].startswith('V'):  # verbs
                word = lemmatizer.lemmatize(each[0], "v")
            elif each[1].startswith('J'):  # adjectives
                word = lemmatizer.lemmatize(each[0], "a")
            elif each[1].startswith('N'):  # nouns
                word = lemmatizer.lemmatize(each[0], "n")
            elif each[1].startswith('R'):  # adverbs
                word = lemmatizer.lemmatize(each[0], "r")
            else:
                word = lemmatizer.lemmatize(each[0])

            # check if token is english word
            en_dict = set([w.lower() for w in words.words('en')])
            names_dict = set([w.lower() for w in names.words()])
            if word.isalpha():
                if wn.synsets(word) or word in en_dict:
                    token = word
                elif word in names_dict:
                    token = "__NAME__"
                else:
                    token = "__NON_WORD__"
            elif word.isdigit():
                token = "__NUM__"
            else:
                token = "__UNK__"

            mapped_word.append(token)
        tokens = mapped_word

    # if check_nonwords is False
    # only map number to __NUM__
    else:
        for index, token in enumerate(tokens):
            if token.isdigit():
                tokens[index] = "__NUM__"

    return tokens


def read_file(path):

    with open(path, errors="ignore") as f:
        data = f.readlines()

    # delete the first line if starts with "subject"
    if "subject" in data[0].lower():
        del data[0]

    # detect a block of forwarded email header
    delete_flag = False
    del_indexes = []
    for index, line in enumerate(data):
        # lowercase all lines
        line = line.lower()
        # indicate the start of header block
        if line.startswith("- - - -") or line.startswith("from :"):
            delete_flag = True
        elif line.startswith("subject"):
            del_indexes.append(index)
            delete_flag = False

        # while delete_flag is True
        # store the index of lines to be deleted
        if delete_flag:
            del_indexes.append(index)

    # delete indexes in reversed order
    for index in sorted(del_indexes, reverse=True):
        del data[index]

    clean_email = ''.join(data)
    # tokenize and remove stopwords
    tokens = tokenize(clean_email)

    return tokens


def get_paths(path):
    file_names = []
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        # check if full path is a file
        if os.path.isfile(full_path):
            file_names.append(file)

    return file_names


def read_dataset(path):
    # list to store lists of email and its label
    emails = []
    # read files in folder
    files_names = get_paths(path)
    for name in tqdm(files_names):
        if "spam" in name:
            label = "spam"
        else:
            label = "ham"

        # read a file from full path
        text = read_file(path + name)
        # append a tuple of tokenized text and its label
        emails.append((text, label))

    return emails

