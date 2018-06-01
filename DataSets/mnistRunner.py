import sys
import mnist
from sklearn.utils import shuffle
import numpy as np
from time import time

from ksu.KSU import KSU

def main(argv=None):

    trainImages, trainLabels = shuffle(mnist.train_images(), mnist.train_labels())
    testImages, testLabels = shuffle(mnist.test_images(), mnist.test_labels())

    train_n = int(0.1 * len(trainLabels))
    test_n = int(0.1 * len(testLabels))

    trainImages = np.reshape(np.array(trainImages[0:train_n,:,:]), [train_n, -1])
    trainLabels = np.array(trainLabels[0:train_n])
    testImages = np.reshape(np.array(testImages[0:test_n,:,:]), [test_n, -1])
    testLabels = np.array(testLabels[0:test_n])

    start = time()
    ksu = KSU(trainImages, trainLabels, 'l2')
    end = time()

    print("Train time: %d", end - start)

    ksuClassifier = ksu.makePredictor(0.1)
    predictedLabels = ksuClassifier.predict(testImages)
    error = np.mean(predictedLabels != testLabels)

    print(error)

if __name__ == '__main__' :
    sys.exit(main())
