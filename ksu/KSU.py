import os
import sys
import logging
import numpy as np
import multiprocessing as mp

from time                     import time
from tqdm                     import tqdm
from tempfile                 import TemporaryFile
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.neighbors.base   import VALID_METRICS
from sklearn.metrics.pairwise import pairwise_distances
from epsilon_net.EpsilonNet   import greedyConstructEpsilonNetWithGram, \
                                     hieracConstructEpsilonNet, \
                                     optimizedHieracConstructEpsilonNet

import Metrics

METRICS = {v:v for v in VALID_METRICS['brute'] if v != 'precomputed'}
METRICS['EditDistance'] = Metrics.editDistance
METRICS['EarthMover']   = Metrics.earthMoverDistance

from Utils import computeGammaSet, \
                  computeLabels, \
                  optimizedComputeAlpha, \
                  computeQ

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
        pass # TODO shoud we also implement this?

    return chosenXs, np.where(chosen)

def compressDataWorker(i, gammaSet, tmpFile, delta, Xs, Ys, metric, gram, minC, maxC, greedy, logLevel=logging.CRITICAL):
    pid         = os.getpid()
    n           = len(Xs)
    numClasses  = len(np.unique(Ys))
    bestGamma   = 0.0
    qMin        = float(np.inf)
    _compression = 0.0
    chosenXs    = None
    chosenYs    = None
    logger      = logging.getLogger('KSU-{}'.format(os.getpid()))

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logLevel)

    logger.debug('PID {} - Choosing from {} gammas'.format(pid, len(gammaSet)))
    for gamma in tqdm(gammaSet):
        tStart = time()
        gammaXs, gammaIdxs = constructGammaNet(Xs, gram, gamma, greedy=greedy)
        compression = float(len(gammaXs)) / n
        logger.debug('PID {p} - Gamma: {g}, net construction took {t:.3f}s, compression: {c}'.format(
            g=gamma,
            t=time() - tStart,
            c=compression,
            p=pid))

        if compression > maxC:
            continue  # heuristic: don't bother compressing by less than an order of magnitude

        if compression < minC:
            break  # heuristic: gammas are increasing, so we might as well stop here

        if len(gammaXs) < numClasses:
            logger.debug(
                'PID {p} - Gamma: {g}, compressed set smaller than number of classes ({cc} vs {c})'
                'no use building a classifier that will never classify some classes'.format(
                    g=gamma,
                    cc=len(gammaXs),
                    c=numClasses,
                    p=pid))
            break

        tStart  = time()
        gammaYs = computeLabels(gammaXs, Xs, Ys, metric)
        logger.debug('PID {p} - Gamma: {g}, label voting took {t:.3f}s'.format(
            g=gamma,
            t=time() - tStart,
            p=pid))

        tStart = time()
        alpha  = optimizedComputeAlpha(gammaYs, Ys, gram[gammaIdxs])
        logger.debug('PID {p} - Gamma: {g}, error approximation took {t:.3f}s, error: {a}'.format(
            g=gamma,
            t=time() - tStart,
            a=alpha,
            p=pid))

        m = len(gammaXs)
        q = computeQ(n, alpha, 2 * m, delta)

        if q < qMin:
            logger.info(
                'PID {p} - Gamma: {g} achieved lowest q so far: {q}, for compression {c}, and empirical error {a}'.format(
                    g=gamma,
                    q=q,
                    c=compression,
                    a=alpha,
                    p=pid))

            qMin        = q
            bestGamma   = gamma
            chosenXs    = gammaXs
            chosenYs    = gammaYs
            _compression = compression

    logger.info('PID {p} - Chosen best gamma: {g}, which achieved q: {q}, and compression: {c}'.format(
        g=bestGamma,
        q=qMin,
        c=_compression,
        p=pid))

    np.savez(tmpFile, X=chosenXs, Y=chosenYs)
    tmpFile.seek(0)
    return qMin, i, _compression, bestGamma

def compressDataWorkerWrapper(outQ, *args, **kwargs):
    try:
        outQ.put(compressDataWorker(*args, **kwargs))
    except:
        outQ.put((float(np.inf), None, None, None,))
        return 1

    return 0

class KSU(object):

    def __init__(self, Xs, Ys, metric, gram=None, prune=False, logLevel=logging.CRITICAL, n_jobs=1):
        self.Xs          = Xs
        self.Ys          = Ys
        self.logger      = logging.getLogger('KSU')
        self.metric      = metric
        self.n_jobs      = n_jobs
        self.chosenXs    = None
        self.chosenYs    = None
        self.compression = None
        self.numClasses  = len(np.unique(self.Ys))
        self.prune       = prune # unused since pruning is not implemented yet

        logging.basicConfig(level=logLevel)

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

        self.gram     = self.gram / np.max(self.gram)

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

    def compressData(self, delta=0.1, minCompress=0.05, maxCompress=0.1, greedy=True, stride=200, logLevel=logging.CRITICAL, numProcs=1):
        """
        Run the KSU algorithm to compress the dataset

        :param delta: confidence for error upper bound
        :param minCompress: minimum compression ratio
        :param maxCompress: maximum compression ratio
        :param greedy: whether to use greedy or hierarchical strategy for net construction
        :param stride: how many gammas to skip between each iteration (similar gammas will produce similar nets)
        :param logLevel: :mod:logging level
        :param numProcs: number of processes to use
        """
        gammaSet = computeGammaSet(self.gram, stride=stride)

        if numProcs == 1:
            tmpFile = TemporaryFile()
            qMin, _, self.compression, bestGamma = \
            compressDataWorker(-1, gammaSet, tmpFile, delta, self.Xs, self.Ys, self.metric, self.gram, minC=minCompress, maxC=maxCompress, greedy=greedy, logLevel=logLevel)
        else:
            if len(gammaSet) % numProcs > 0:
                gammaSet = gammaSet[:-(len(gammaSet) % numProcs)]

            # mp.log_to_stderr(logging.DEBUG) # uncomment to debug concurrency

            gammaSets = np.reshape(gammaSet, [numProcs, -1])
            outputQ   = mp.Queue()
            tmpFiles  = [TemporaryFile() for _ in range(numProcs)]
            procs     = [mp.Process(target=compressDataWorkerWrapper,
                                    args=(outputQ, i, gammaSets[i], tmpFiles[i], delta, self.Xs, self.Ys, self.metric, self.gram, minCompress, maxCompress, greedy,),
                                    kwargs={'logLevel': logLevel})
                         for i in range(numProcs)]

            for p in procs:
                p.start()

            for p in procs:
                p.join()

            results = sorted([outputQ.get() for _ in procs], key=lambda r: r[0]) # sorted by qMin
            qMin, i, self.compression, bestGamma = results[0]
            tmpFile = tmpFiles[i]

        if qMin == float(np.inf):
            self.logger.critical('No gamma was chosen! check logs')
            return

        chosen        = np.load(tmpFile)
        self.chosenXs = chosen['X']
        self.chosenYs = chosen['Y']

        self.logger.info('Chosen best gamma: {g}, which achieved q: {q}, and compression: {c}'.format(
            g=bestGamma,
            q=qMin,
            c=self.compression))

