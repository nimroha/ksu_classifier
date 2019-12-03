import os
import sys
import logging
import numpy as np
import multiprocessing as mp

from time                     import time
from tqdm                     import tqdm
from tempfile                 import NamedTemporaryFile
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.neighbors.base   import VALID_METRICS
from sklearn.metrics.pairwise import pairwise_distances


import ksu.Metrics
from ksu.epsilon_net.EpsilonNet import greedyConstructEpsilonNetWithGram, \
                                       optimizedHieracConstructEpsilonNet
from ksu.Utils import computeGammaSet, \
                      computeLabels, \
                      optimizedComputeAlpha, \
                      computeQ, \
                      configLogger

METRICS = {v:v for v in VALID_METRICS['brute'] if v != 'precomputed'}
METRICS['EditDistance'] = ksu.Metrics.editDistance
METRICS['EarthMover']   = ksu.Metrics.earthMoverDistance


def constructGammaNet(Xs, gram, gamma, prune=False, greedy=True):
    """
    Construct an epsilon net for parameter gamma

    :param Xs: points
    :param gram: gram matrix of the points
    :param gamma: net parameter
    :param prune: whether to prune the net after construction
    :param greedy: whether to build the net greedily or with an hierarchical strategy

    :return: the chosen points and their indices
    """
    if greedy:
        chosenXs, chosen = greedyConstructEpsilonNetWithGram(Xs, gram, gamma)
    else:
        chosenXs, chosen = optimizedHieracConstructEpsilonNet(Xs, gram, gamma)

    if prune:
        pass # TODO should we also implement this?

    return chosenXs, np.where(chosen)

def compressDataWorker(i, gammaSet, tmpFile, delta, Xs, Ys, metric, gram, minC, maxC, greedy, logLevel=logging.CRITICAL):
    pid         = os.getpid()
    n           = len(Xs)
    numClasses  = len(np.unique(Ys))
    bestGamma   = 0.0
    qMin        = float(np.inf)
    bestCompres = 0.0
    chosenXs    = None
    chosenYs    = None
    logger      = logging.getLogger('KSU')

    logger.debug('Choosing from {} gammas'.format(len(gammaSet)))
    for gamma in tqdm(gammaSet):
        tStart = time()
        gammaXs, gammaIdxs = constructGammaNet(Xs, gram, gamma, greedy=greedy)
        compression = float(len(gammaXs)) / n
        logger.debug('Gamma: {g}, net construction took {t:.3f}s, compression: {c}'.format(
            g=gamma,
            t=time() - tStart,
            c=compression))

        if compression > maxC:
            continue  # heuristic: don't bother compressing by less than an order of magnitude

        if compression < minC:
            break  # heuristic: gammas are increasing, so we might as well stop here

        if len(gammaXs) < numClasses:
            logger.debug(
                'Gamma: {g}, compressed set smaller than number of classes ({cc} vs {c})'
                'no use building a classifier that will never classify some classes'.format(
                    g=gamma,
                    cc=len(gammaXs),
                    c=numClasses))
            break

        tStart  = time()
        gammaYs = computeLabels(gammaXs, Xs, Ys, metric)
        logger.debug('Gamma: {g}, label voting took {t:.3f}s'.format(
            g=gamma,
            t=time() - tStart))

        tStart = time()
        alpha  = optimizedComputeAlpha(gammaYs, Ys, gram[gammaIdxs])
        logger.debug('Gamma: {g}, error approximation took {t:.3f}s, error: {a}'.format(
            g=gamma,
            t=time() - tStart,
            a=alpha))

        m = len(gammaXs)
        q = computeQ(n, 2 * m, alpha, delta)

        if q < qMin:
            logger.info(
                'Gamma: {g} achieved lowest q so far: {q}, for compression {c}, and empirical error {a}'.format(
                    g=gamma,
                    q=q,
                    c=compression,
                    a=alpha))

            qMin        = q
            bestGamma   = gamma
            chosenXs    = gammaXs
            chosenYs    = gammaYs
            bestCompres = compression

    if qMin == float(np.inf):
        logger.info('No good gamma found between gammas: [{min} ... {max}]'.format(
            p=pid,
            min=gammaSet.min(),
            max=gammaSet.max()))
        return qMin, i, bestCompres, bestGamma

    logger.info('Chosen best gamma: {g}, which achieved q: {q}, and compression: {c}'.format(
        g=bestGamma,
        q=qMin,
        c=bestCompres))

    np.savez(tmpFile, X=chosenXs, Y=chosenYs)

    return qMin, i, bestCompres, bestGamma

class KSU(object):

    def __init__(self, Xs, Ys, metric, gram=None, prune=False, logLevel=logging.CRITICAL, n_jobs=1):
        self.Xs          = Xs
        self.Ys          = Ys
        self.metric      = metric
        self.n_jobs      = n_jobs
        self.chosenXs    = None
        self.chosenYs    = None
        self.compression = None
        self.numClasses  = len(np.unique(self.Ys))
        self.prune       = prune # unused since pruning is not implemented yet

        self.logger = logging.getLogger('KSU')
        configLogger(self.logger, logLevel)

        if isinstance(metric, str) and metric not in METRICS.keys():
            raise RuntimeError(
                '"{m}" is not a built-in metric. use one of'
                '{ms}'
                'or provide your own custom metric as a callable'.format(
                    m=metric,
                    ms=METRICS.keys()))

        if gram is None:
            self.logger.info('Computing Gram matrix...')
            tStartGram = time()
            self.gram  = pairwise_distances(self.Xs, metric=self.metric, n_jobs=self.n_jobs)
            self.logger.debug('Gram computation took {:.3f}s'.format(time() - tStartGram))
        else:
            self.gram = gram

        self.gram = self.gram / np.max(self.gram)

    def getCompressedSet(self):
        """
        Getter for compressed set

        :return: the compressed set and its labels

        :raise: :class:RuntimeError if :func:KSU.KSU.compressData was not run before
        """
        if self.chosenXs is None:
            raise RuntimeError('getCompressedSet - you must run KSU.compressData first')

        return self.chosenXs, self.chosenYs

    def getCompression(self):
        """
        Getter for compression ratio

        :return: the compression ratio

        :raise: :class:RuntimeError if :func:KSU.KSU.compressData was not run before
        """
        if self.compression is None:
            raise RuntimeError('getCompression - you must run KSU.compressData first')

        return self.compression

    def getClassifier(self):
        """
        Getter for 1-NN classifier fitted on the compressed set

        :return: an :mod:sklearn.KNeighborsClassifier instance

        :raise: :class:RuntimeError if :func:KSU.KSU.compressData was not run before
        """
        if self.chosenXs is None:
            raise RuntimeError('getClassifier - you must run KSU.compressData first')

        h = KNeighborsClassifier(n_neighbors=1, metric=self.metric, algorithm='auto', n_jobs=self.n_jobs)
        h.fit(self.chosenXs, self.chosenYs)

        return h

    def compressData(self, delta=0.1, minCompress=0.01, maxCompress=0.1, greedy=True, stride=200, logLevel=logging.CRITICAL):
        """
        Run the KSU algorithm to compress the dataset

        :param delta: confidence for error upper bound
        :param minCompress: minimum compression ratio
        :param maxCompress: maximum compression ratio
        :param greedy: whether to use greedy or hierarchical strategy for net construction
        :param stride: how many gammas to skip between each iteration (similar gammas will produce similar nets)
        :param logLevel: :mod:logging level
        """
        gammaSet = computeGammaSet(self.gram, stride=stride) # TODO add heuristic to throw away gammas that won't satisfy the compression limits
        numProcs = self.n_jobs if self.n_jobs > 0 else mp.cpu_count() + 1 + self.n_jobs
        self.logger.info('using {n} cores to process {g} gammas'.format(n=numProcs, g=len(gammaSet)))

        if numProcs == 1:
            tmpFile = NamedTemporaryFile()
            qMin, _, self.compression, bestGamma = \
            compressDataWorker(-1, gammaSet, tmpFile.name, delta, self.Xs, self.Ys, self.metric, self.gram, minC=minCompress, maxC=maxCompress, greedy=greedy, logLevel=logLevel)
        else:
            numJobs = 4 * numProcs
            if len(gammaSet) % numJobs > 0:
                gammaSet = gammaSet[len(gammaSet) % numJobs:]

            # mp.log_to_stderr(logging.DEBUG) # uncomment to debug concurrency

            tmpFiles  = [NamedTemporaryFile() for _ in range(numJobs)]
            gammaSets = np.reshape(gammaSet, [numJobs, -1])
            pool      = mp.Pool(numProcs) # TODO send Xs, Ys and gram in shared memory
            results   = [pool.apply_async(func=compressDataWorker,
                                          args=(i, gammaSets[i], tmpFiles[i].name, delta, self.Xs, self.Ys, self.metric, self.gram, minCompress, maxCompress, greedy,),
                                          kwds={'logLevel': logLevel}) for i in range(numJobs)]
            pool.close()
            pool.join()

            results = sorted([r.get() for r in results], key=lambda r: r[0])  # sorted by qMin
            qMin, i, self.compression, bestGamma = results[0]
            tmpFile = tmpFiles[i].name if i is not None else 0

        if qMin == float(np.inf):
            self.logger.critical('No gamma was chosen! check logs')
            return

        chosen        = np.load(tmpFile, allow_pickle=True)
        self.chosenXs = chosen['X']
        self.chosenYs = chosen['Y']

        self.logger.info('Chosen best gamma: {g}, which achieved q: {q}, and compression: {c}'.format(
            g=bestGamma,
            q=qMin,
            c=self.compression))

