import os
import re

"""

This file defines all files and directories needed in computations.

my_dir              - directory containing all the individual .py files (i.e. sentiment_class, naive_bayes,
                        sent_score, test_sent and this file, i.e. user_file)
dict_file           - .txt word sentiment vocabulary, currently AFINN-111 and opinionlexiconEnglish
                        are implemented. Should be saved in the Dict_files subfolder.
                        AFINN-111: http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=6010
                        opinionlexiconEnglish: https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
train_file          - .txt file containing train Factiva articles
test_file           - .txt file containing test Factiva articles
length_test_file    - number of articles in the test file
version_train       - version of the articles in the train file. Currently implemented is The Wall Street
                        Journal and Reuters ('WSJ', 'RET')
version_test        - version of the articles in the test file. Currently implemented is The Wall Street
                        Journal and Reuters ('WSJ', 'RET')
crossvalidation     - ratio of files in training set used for randomized crossvalidation
code_test_set       - .txt file containing articles used for tests (pytest module is used)

Files train_sentiment_file should be composed of article scores corresponding to the order
of articles in the train_file.

"""

my_dir = r'D:\Materials\Programming\Python\Ksenia_master'
os.chdir(my_dir)
dict_file = 'AFINN-111'
train_file = r'Input_files\Factiva500.txt'
test_file = r'Input_files\FactivaRest.txt'
with open(r'Input_files\train_sentiment_file.csv', 'r') as doc:
    file = doc.read()
train_sentiment_file = re.split('\\n', file)[:-1]
length_test_file = 10000
test_sentiment_file = [0] * length_test_file
version_train = 'RET'
version_test = 'RET'
crossvalidation = 0.1
code_test_set = r'Input_files\TstFl.txt'

