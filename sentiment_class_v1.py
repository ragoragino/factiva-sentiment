import re
import os
import sys
import collections as co
import random as rand
import weakref
import inspect
import user_file as uf
os.chdir(uf.my_dir)

"""
This file implements the processing of the Factiva articles.
"""


class SentimentError(Exception):
    """
    SentimentError occurs when no proper dictionary sentiment file is selected
    """
    def __init__(self):
        super().__init__(self, 'You must select an appropriate name of the sentiment file!')


class VersionError(Exception):
    """
    VersionError occurs when no proper version of the news source is selected
    """
    def __init__(self):
        super().__init__(self, 'You must select the correct version of the news!')


def sentiment_dictionary(dict_type='AFINN-111'):
    """
    Function loading sentiment dictionary

    Args:
        dict_type (str): The type of dictionary, currently
        implemented are options AFINN-111 or opinionlexiconEnglish

    Returns:
        dict: Dictionary with words as keys and sentiment
        scores as values

    Raises:
        SentimentError in case unrecognized dictionary is selected
    """
    if dict_type == 'AFINN-111':
        sentlist = [re.split('\t|\n|', line) for line in open(r'Dict_files\AFINN\AFINN-111.txt', encoding='UTF-8')]
        sentdict = {line[0]: line[1] for line in sentlist}
    elif dict_type == 'opinionlexiconEnglish':
        with open(r'Dict_files\opinionlexiconEnglish\negative-words.txt', 'r', encoding='UTF-8') as doc:
            file1 = doc.read()
        sentdict = {i: -1 for i in file1.split('\n') if i != ''}
        with open(r'Dict_files\opinionlexiconEnglish\positive-words.txt', 'r', encoding='UTF-8') as doc:
            file2 = doc.read()
        for i in file2.split('\n'):
            if i != '':
                sentdict.update({i: 1})
        sentdict.__delitem__('bull****')
        sentdict.__delitem__('bull----')
        sentdict.__delitem__('f**k')
    else:
        raise SentimentError()
    return sentdict


class TrainArticle:
    """
    Class for articles used in training
    """

    def __init__(self, date, newspaper, article, words, sentiment):
        """

        Args:
            date (str): The date of the publication of the article, of the format '%d %B %Y', e.g. '15 March 2017'
            newspaper (str): The newspaper where the article was published
            article (str): The text of the article
            words (str): Number of words in the article
            sentiment (int): Sentiment score for the article
        """
        self.date = date
        self.worddict = co.Counter()
        self.newspaper = newspaper
        self.article = article
        self.words = int(''.join(re.split(',', words)))
        self.sentiment = sentiment

    def senteval(self, sent_dictionary):
        """
        Function counting the words from dictionary files present in the article

        Args:
            sent_dictionary (dict): A sentiment dictionary object returned
            by the function sentiment_dicionary

        Returns:
            appends sentvalue attribute, i.e. an aggregate score for article,
            to the article object and updates the worddict co.Counter dictionary

        """
        self.sentvalue = 0
        for key, value in sent_dictionary.items():
            # Only exact matches, e.g. 'ill' does not match neither 'will' nor 'illness',
            # but matches any preceding/following non-alphanumeric character, e.g. 'money,'
            articlematch = re.findall('\W' + '(' + key + ')' + '\W', self.article)
            self.worddict[(key, self.sentiment)] = len(articlematch)
            self.sentvalue += len(articlematch) * int(value)


class TestArticle:
    """
    Class used to construct objects useful for testing. The main feature
    is that it pools TrainArticle objects with identical dates
    into one object.

    Attributes:
        _testdict (weakref.WeakValueDictionary): dictionary holding weak references
         to the existing objects of the class as values and individual dates as keys
    """
    _testdict = weakref.WeakValueDictionary()

    def __new__(cls, date, worddict):
        """

        Args:
            date (str): date when the article was published, of the format '%d %B %Y', e.g. '15 March 2017'
            worddict (co.Counter): co.Counter() dictionary from TrainArticle object

        Returns:
            a constructed object only if the object with that date does not already exist. If it does,
            the __new__ method returns the existing object
        """
        newobject = cls._testdict.get(date)
        if not newobject:
            newobject = super().__new__(cls)
            cls._testdict[date] = newobject
        return newobject

    def __init__(self, date, worddict):
        """

        Args:
            date (str): date when the article was published, of the format '%d %B %Y', e.g. '15 March 2017'
            worddict (co.Counter): co.Counter() dictionary from TrainArticle object
        """
        if not hasattr(self, 'init'):
            self.date = date
            self.worddict = co.Counter()
            self.worddict += worddict
            self.init = True
        else:
            self.worddict += worddict


def artscrape(filename, sent_dictionary, sentiment, version):
    """
    This function processes the whole Factiva file

    Args:
        filename (str): the name of the Factiva file in the current working directory
        sent_dictionary (dict): dictionary returned by sentiment_dictionary function
        sentiment (list): list of sentiment scores matching the order of article present in the Factiva file
        version (str): Source of the news, currently implemented options are Wall Street Journal (WSJ)
        and Reuters (RET)

    Returns:
        List with initialized TrainArticle objects

    Raises:
        VersionError if the unrecognized version is selected.

    """
    if version == 'WSJ':
        articlesearch = re.compile('(?<=\\n\\n\s\s\sLP\\t)(.)*?(?=\\n\\n\s\s\sCO\\t)', re.DOTALL)
        datesearch = re.compile('(?<=\\n\s\s\sPD\\t)(.)+(?=\\n)')
        newssearch = re.compile('(?<=\\n\s\s\sSN\\t)(.)+(?=\\n)')
        wordssearch = re.compile('(?<=\\n\s\s\sWC\\t)([0-9,]+)')
    elif version == 'RET':
        articlesearch = re.compile('(?<=\\n\sLP\\t)(.)+?(?=\\n\\n\s+RF\\t)', re.DOTALL)
        datesearch = re.compile('(?<=\\n\sPD\\t)(.)+(?=\\n)')
        newssearch = re.compile('(?<=\\n\sSN\\t)(.)+(?=\\n)')
        wordssearch = re.compile('(?<=\\n\sWC\\t)([0-9,]+)')
    else:
        raise VersionError()
    with open(filename, encoding='utf-8') as doc:
        text = doc.read()
    position = 0
    articlebox = []
    while True:
        try:
            datematch = datesearch.search(text, pos=position)
            newsmatch = newssearch.search(text, pos=position)
            artimatch = articlesearch.search(text, pos=position)
            wordsmatch = wordssearch.search(text, pos=position)
            newspiece = TrainArticle(datematch.group(), newsmatch.group(), artimatch.group(),
                                     wordsmatch.group(), sentiment.pop(0))
            # plus 1 needed for the cases of empty articles when we want the position to move
            position = artimatch.end() + 1
            newspiece.senteval(sent_dictionary)
            articlebox.append(newspiece)
        except (AttributeError, IndexError):
            break
    return articlebox


if __name__ == '__main__' \
        or inspect.stack()[-1].filename == r'D:\Materials\Programming\Python\Ksenia_master\master_file.py'\
        or inspect.stack()[-1].filename == r'D:\Materials\Programming\Python\Ksenia_master\sent_score.py':
    # The part above checks who is the caller.
    # This part is mainly constructed for optimization of test procedures (that fall in the else category)
    sentdict = sentiment_dictionary(uf.dict_file)
    articlebox = artscrape(uf.train_file, sentdict, uf.train_sentiment_file, uf.version_train)
    testbox = artscrape(uf.test_file, sentdict, uf.test_sentiment_file, uf.version_test)
    testset = {TestArticle(i.date, i.worddict) for i in testbox}
else:
    pass

