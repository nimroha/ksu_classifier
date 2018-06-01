import sys
from mnist import MNIST
from sklearn.utils import shuffle
import numpy as np
import logging
from time import time

from ksu.KSU import KSU

def main(argv=None):

    mnist_dir = 'C:\Users\yuvalnissan\Desktop\MNIST'
    mndata = MNIST(mnist_dir)
    mndata.gz = True
    mndata.load_training()
    mndata.load_testing()

    train, trainLabels = shuffle(mndata.train_images, mndata.train_labels)
    test, testLabels = shuffle(mndata.test_images, mndata.test_labels)

    train = np.array(train[:9999])
    trainLabels = np.array(trainLabels[:9999])
    test = np.array(test[:999])
    testLabels = np.array(testLabels[:999])

    start = time()
    ksu = KSU(train, trainLabels, 'l2')
    end = time()

    print("Train time: %d", end - start)

    ksuClassifier = ksu.makePredictor(0.1)
    predictedLabels = ksuClassifier.predict(test)
    error = np.mean(predictedLabels != testLabels)

    print(error)

if __name__ == '__main__' :
    sys.exit(main())
