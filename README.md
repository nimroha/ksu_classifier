## KSU Compression Algorithm Implementation ##

Algortihm 1 from [Nearest-Neighbor Sample Compression: Efficiency, Consistency, Infinite Dimensions](https://arxiv.org/abs/1705.08184)

Installation
------------
* With pip: `pip install ksu`
* From source:
    * `git clone --recursive https://github.com/nimroha/ksu_classifier.git`
    * `cd ksu_classifier`
    * `python setup.py install`
    
 Usage
 -----
 This package provides a class `KSU(Xs, Ys, metric, [gramPath, prune])`
 
 `Xs` and `Ys` are the data points and their respective labels as [numpy  arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) 
 
 `metric` is either a callable to compute the metric or a string that names one of our provided metrics (print `ksu.KSU.METRICS.keys()` for the full list) 
 
 `gramPath` _(optional, default=None)_ a path to a precomputed [gramian matrix](http://mathworld.wolfram.com/GramMatrix.html)
 
 `prune` _(optional, default=False)_ a boolean indicating whether to prune the compressed set or not (Algorithm 2 from [Near-optimal sample compression for nearest neighbors](https://arxiv.org/abs/1404.3368))
 
  <br>
 
  `KSU` provides a method `makePredictor([delta])`
  
  Which returns a 1-NN Classifer (based on [sklearn's K-NN](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)) fitted to the compressed data, where `delta` _(optional, default=5%)_ is the required confidence of said classifier.