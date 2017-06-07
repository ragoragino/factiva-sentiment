import time
starttime = time.time()
import sys
import os
import collections as co
import random as rand
import copy
import logging
import user_file as uf
import sentiment_class_v1 as sc
import sent_score as ss
import naive_bayes as nb
import logging
os.chdir(uf.my_dir)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    error, prior, likelihood = nb.bayesprediction(sc.articlebox, crossvalidation=uf.crossvalidation)
    test_set = [i for i in sc.testset]
    test_index = range(len(test_set))
    nb.testbayes(test_set, test_index, prior, likelihood)
    with open(r'Output_files\test_score.csv', 'w') as doc:
        for i in test_set:
            doc.write(i.date + ', ' + i.decision + '\n')
    ss.sentwrite(sc.articlebox + sc.testbox)
    avg_word_stat = [sum(i.worddict.values()) for i in sc.articlebox]
    logger.info("Average number of words found in an article: {} \n".format(sum(avg_word_stat)/len(avg_word_stat)))
    logger.info("The training accuracy is: {} \n".format(error))
    # logger.info(likelihood)
    # logger.info(prior)
    logger.info("The running time of the code is: {} s \n".format(time.time() - starttime))
