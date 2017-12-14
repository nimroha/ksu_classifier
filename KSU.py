

class KSU(object):

    def __init__(self, compressedSet):
        self.set = compressedSet

    def predict(self, example):
        raise NotImplementedError("prediction not implemented")