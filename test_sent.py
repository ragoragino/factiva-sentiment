import os
import collections as co
import math
import pytest
import sentiment_class_v1 as sc
import naive_bayes as nb

"""
This is a file for testing, using pytest module.
"""

my_dir = r'D:\Materials\Programming\Python\'
os.chdir(my_dir)

sentiment_dictionary = {'market': -1, 'money': 1, 'growth': 1, 'tax': -1}
filename = r'Input_files\TstFl.txt'
tdict = {'RET': {'words': 132, 'date': '19 March 2017', 'newspaper': 'Reuters News', 'sentvalue': 1,
                     'worddict': co.Counter({('market', 0): 1, ('growth', 0): 1, ('money', 0): 1, ('tax', 0): 0})},
         'WSJ': {'words': 918, 'date': '1 May 2017', 'newspaper': 'The Wall Street Journal Online', 'sentvalue': -1,
                     'worddict': co.Counter({('market', 1): 2, ('growth', 1): 0, ('money', 1): 2, ('tax', 1): 1})}}

@pytest.fixture
def ret_scrape():
    return sc.artscrape(filename, sentiment_dictionary, [0], 'RET')

@pytest.fixture
def wsj_scrape():
    return sc.artscrape(filename, sentiment_dictionary, [1], 'WSJ')


class TestSentimentDict:
    def test_sentiment_dictionary_default(self):
        sent_load = sc.sentiment_dictionary()
        assert len(sent_load) == 2477

    def test_sentiment_dictionary_afinn(self):
        sent_load = sc.sentiment_dictionary(r'AFINN-111')
        assert len(sent_load) == 2477

    def test_sentiment_dictionary_ole(self):
        sent_load = sc.sentiment_dictionary('opinionlexiconEnglish')
        assert len(sent_load) == 6780

    def test_sentiment_dictionary_fault(self):
        try:
            sc.sentiment_dictionary('wrongdictionaryfile')
        except sc.SentimentError:
            assert True
        else:
            assert False


class TestArtscrape:
    def test_article_date_ret(self, ret_scrape):
        trainarticlevalue = getattr(*ret_scrape, 'date')
        testobjectvalue = tdict['RET']['date']
        assert trainarticlevalue == testobjectvalue

    def test_article_date_wsj(self, wsj_scrape):
        trainarticlevalue = getattr(*wsj_scrape, 'date')
        testobjectvalue = tdict['WSJ']['date']
        assert trainarticlevalue == testobjectvalue

    def test_article_words_ret(self, ret_scrape):
        trainarticlevalue = getattr(*ret_scrape, 'words')
        testobjectvalue = tdict['RET']['words']
        assert trainarticlevalue == testobjectvalue

    def test_article_words_wsj(self, wsj_scrape):
        trainarticlevalue = getattr(*wsj_scrape, 'words')
        testobjectvalue = tdict['WSJ']['words']
        assert trainarticlevalue == testobjectvalue

    def test_article_worddict_ret(self, ret_scrape):
        trainarticlevalue = getattr(*ret_scrape, 'worddict')
        testobjectvalue = tdict['RET']['worddict']
        assert trainarticlevalue == testobjectvalue

    def test_article_worddict_wsj(self, wsj_scrape):
        trainarticlevalue = getattr(*wsj_scrape, 'worddict')
        testobjectvalue = tdict['WSJ']['worddict']
        assert trainarticlevalue == testobjectvalue

    def test_article_sentvalue_ret(self, ret_scrape):
        trainarticlevalue = getattr(*ret_scrape, 'sentvalue')
        testobjectvalue = tdict['RET']['sentvalue']
        assert trainarticlevalue == testobjectvalue

    def test_article_sentvalue_wsj(self, wsj_scrape):
        trainarticlevalue = getattr(*wsj_scrape, 'sentvalue')
        testobjectvalue = tdict['WSJ']['sentvalue']
        assert trainarticlevalue == testobjectvalue

    def test_article_newspaper_ret(self, ret_scrape):
        trainarticlevalue = getattr(*ret_scrape, 'newspaper')
        testobjectvalue = tdict['RET']['newspaper']
        assert trainarticlevalue == testobjectvalue

    def test_article_newspaper_wsj(self, wsj_scrape):
        trainarticlevalue = getattr(*wsj_scrape, 'newspaper')
        testobjectvalue = tdict['WSJ']['newspaper']
        assert trainarticlevalue == testobjectvalue

    def test_version_error(self):
        try:
            sc.artscrape(filename, sentiment_dictionary, [1], 'WRONGVERSION')
        except sc.VersionError:
            assert True
        else:
            assert False


class TestNaiveBayes:
    def test_prior(self, ret_scrape, wsj_scrape):
        priordict = nb.prior(ret_scrape + wsj_scrape)
        priortest = math.log(1/2)
        priortestdict = {0: priortest, 1: priortest}
        assert priortestdict == priordict

    def test_likelihood(self, ret_scrape, wsj_scrape):
        likelihooddir = nb.likelihood(ret_scrape + wsj_scrape)
        classcardinality = co.Counter({0: 3, 1: 5})
        cardinality = 4
        likelihoodtestdict = co.Counter({('market', 0): math.log((1 + 1)/(cardinality + classcardinality[0])),
                                  ('growth', 0): math.log((1 + 1)/(cardinality + classcardinality[0])),
                                  ('money', 0): math.log((1 + 1)/(cardinality + classcardinality[0])),
                                  ('tax', 0): math.log(1/(cardinality + classcardinality[0])),
                                  ('market', 1): math.log((2 + 1)/(cardinality + classcardinality[1])),
                                  ('money', 1): math.log((2 + 1)/(cardinality + classcardinality[1])),
                                  ('tax', 1): math.log((1 + 1)/(cardinality + classcardinality[1])),
                                  ('growth', 1): math.log(1/(cardinality + classcardinality[1]))})
        assert likelihoodtestdict == likelihooddir

    def test_testbayes(self, ret_scrape, wsj_scrape):
        nb.testbayes(ret_scrape + wsj_scrape, [0, 1], nb.prior(ret_scrape + wsj_scrape),
                     nb.likelihood(ret_scrape + wsj_scrape))
        assert round(ret_scrape[0].posterior[0], 10) == round(math.log(4/7**3), 10)
        assert round(ret_scrape[0].posterior[1], 10) == round(math.log(1/(2*81)), 10)
        assert round(wsj_scrape[0].posterior[0], 10) == round(math.log(2**3/7**5), 10)
        assert round(wsj_scrape[0].posterior[1], 10) == round(math.log(1/9**3), 10)

