import sys
import mnist
import numpy as np

from sklearn.utils import shuffle
from time          import time

from ksu.KSU import KSU
from ksu.Metrics import makeLn

def main(argv=None):

    trainImages, trainLabels = shuffle(mnist.train_images(), mnist.train_labels())
    testImages, testLabels   = shuffle(mnist.test_images(), mnist.test_labels())

    train_n = int(0.1 * len(trainLabels))
    test_n  = int(0.1 * len(testLabels))

    trainImages = np.reshape(np.array(trainImages[0:train_n,:,:]), [train_n, -1])
    trainLabels = np.array(trainLabels[0:train_n])
    testImages  = np.reshape(np.array(testImages[0:test_n,:,:]), [test_n, -1])
    testLabels  = np.array(testLabels[0:test_n])

    startAll = start = time()
    ksu = KSU(trainImages, trainLabels, 'l2')
    print("Init time: {:.3f}".format(time() - start))

    start = time()
    ksu.compressData(0.1)
    print('Compress time: {:.3f}'.format(time() - start))

    start         = time()
    ksuClassifier = ksu.getClassifier()
    print('Fit time: {:.3f}'.format(time() - start))

    start           = time()
    predictedLabels = ksuClassifier.predict(testImages)
    print('Predict time: {:.3f}'.format(time() - start))

    error = np.mean(predictedLabels != testLabels)

    print('error: {}  total runtime: {:.3f}'.format(error, time() - startAll))

if __name__ == '__main__' :
    sys.exit(main())
