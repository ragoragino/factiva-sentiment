import time
import datetime as dt
import user_file as uf

"""
This file saves output from simple sentiment averaging.
"""


def sentwrite(articlebox):
    """
    Function that calculates the average sentiment score for a day.

    Args:
        articlebox (list): list of TrainArticle objects returned by the function artscrape

    Returns:
        Saves .csv file with dates and corresponding sentiment scores

    """
    mapdict = {i.date: 0 for i in articlebox}
    for i in articlebox:
        mapdict[i.date] += i.sentvalue / i.words
    sortdates = sorted([(dt.datetime.strptime(i, "%d %B %Y").date(), j) for i, j in mapdict.items()])
    begin = sortdates[0][0]
    end = sortdates[-1][0]
    delta = (end-begin).days
    dates = [begin + dt.timedelta(days=x) for x in range(0, delta + 1)]
    y = ['NA'] * len(dates)
    for i, j in sortdates:
        if i in dates:
            y[dates.index(i)] = j
    datesclass = []
    for i in dates:
        datesclass.append(dt.datetime.strftime(i, "%d %B %Y"))
    with open(r'Output_files\sentiment_score.csv', 'w') as doc:
        for i, j in zip(datesclass, y):
            doc.write(i + ',' + str(j) + '\n')