import math
import collections as co
import random as rand
import copy

"""
This file implements the computation of the Naive Bayes algorithm.
"""


def prior(trainsample):
    """
    Function that computes prior of a sample of articles

    Args:
        trainsample (list): list of TrainArticle() objects from the sentiment_class_v1 module

    Returns:
        dictionary with classes as keys and their priors as values
    """
    counter = co.Counter()
    for i in trainsample:
        counter[i.sentiment] += 1
    sumcounter = sum(counter.values())
    priordict = {i: math.log(j/sumcounter) for i, j in counter.items()}
    return priordict


def likelihood(trainsample):
    """
    Function that computes likelihood of a sample of articles

    Args:
        trainsample (list): list of TrainArticle() objects from the sentiment_class_v1 module

    Returns:
        dictionary with (word, class) tuple as a key and the likelihood P(w|c) as a value
    """
    wordclass = co.Counter()
    for i in trainsample:
        wordclass += i.worddict
    classcardinality = co.Counter()
    [classcardinality.update({i[1]: j}) for i, j in wordclass.items()]
    individualwords = {i[0] for i, j in wordclass.items()}
    cardinality = len(individualwords)
    classes = {i for i, j in classcardinality.items()}
    likelihoodvar = {}
    for i in individualwords:
        for j in classes:
            try:
                likelihoodvar[(i, j)] = math.log((wordclass[(i, j)] + 1)/(classcardinality[j] + cardinality))
            except KeyError:
                likelihoodvar[(i, j)] = math.log(1/(classcardinality[j] + cardinality))
    return likelihoodvar


def bayesprediction(dataset, crossvalidation):
    """
    Function that implements randomized cross-validation and calculates
    the final average error rate and prior and likelihood on the train sample

    Args:
        dataset (list): list of TrainArticle() objects from the sentiment_class_v1 module
        crossvalidation: the ratio of test sample used for cross-validation

    Returns:
        errorrate (float) of the cross-validated classification,
        finalprior (dictionary) and finallikelihood (dictionary) for the whole training file
    """
    samplelength = len(dataset)
    samplepos = rand.sample(range(samplelength), samplelength)
    steplength = int(crossvalidation * samplelength)
    noiterations = int(samplelength / steplength)
    for i in range(noiterations):
        sampleposcopy = copy.deepcopy(samplepos)
        testpos = [sampleposcopy[j] for j in range(i, samplelength, noiterations)]
        trainsample = [dataset[j] for j in sampleposcopy if j not in testpos]
        priorvar = prior(trainsample)
        likelihoodvar = likelihood(trainsample)
        testbayes(dataset, testpos, priorvar, likelihoodvar)
    errorrate = sum([i.decision == i.sentiment for i in dataset])/samplelength
    finalprior = prior(dataset)
    finallikelihood = likelihood(dataset)
    return errorrate, finalprior, finallikelihood


def testbayes(dataset, index, priorvar, likelihoodvar):
    """
    Function that calculates the likelihood of combinations of different classes for the test sample

    Args:
        dataset (list): list of TrainArticle() or TestArticle() objects
            from the sentiment_class_v1 module
        index (list or range): index of values which are classified according to this algorithm, i.e. in the case
            when not all objects in dataset are being classified
        priorvar (dict): dictionary returned from prior function
        likelihoodvar (dict): dictionary returned from likelihood function

    Returns:
        appends posterior probability attribute to each dataset object, i.e. P(c|w),
        and also final decision that is the maximization procedure over different c,
        i.e. sentiment classes
    """
    classes = {i for i, j in priorvar.items()}
    for j in index:
        dataset[j].posterior = {}
        for i in classes:
            newvalue = priorvar[i]
            for x, y in dataset[j].worddict.items():
                try:
                    newvalue += likelihoodvar[(x[0], i)] * y  # Adding the count of words
                except KeyError:
                    pass
            dataset[j].posterior[i] = newvalue
        dataset[j].decision = max(dataset[j].posterior, key=dataset[j].posterior.get)
    return


