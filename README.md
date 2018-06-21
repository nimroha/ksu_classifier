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
 This package provides a class `KSU(Xs, Ys, metric, [gram, prune, logLevel, n_jobs])`
 
 `Xs` and `Ys` are the data points and their respective labels as [numpy  arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) 
 
 `metric` is either a callable to compute the metric or a string that names one of our provided metrics (print `ksu.KSU.METRICS.keys()` for the full list) 
 
 `gram` _(optional, default=None)_ a precomputed [gramian matrix](http://mathworld.wolfram.com/GramMatrix.html), will be calculated if not provided.
 
 `prune` _(optional, default=False)_ a boolean indicating whether to prune the compressed set or not (Algorithm 2 from [Near-optimal sample compression for nearest neighbors](https://arxiv.org/abs/1404.3368))

 `logLevel _(optional, default='CRITICAL')_ a string indicating the logging level (set to 'INFO' or 'DEBUG' to get more information)

 `n_jobs` _(optional, default=1)_ an integer defining how many cpus to use, pass -1 to use all. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
 
  <br>
 
  `KSU` provides a method `compressData([delta])`
  
  Which selects the subset with the lowest estimated error with confidence `1 - delta`.
  
  You can then run `getClassifier()` which returns a 1-NN Classifer (based on [sklearn's K-NN](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)) fitted to the compressed data.
  
  Or, run `getCompressedSet()` to get the compressed data as a tuple of numpy arrays `(compressedXs, compressedYs)`.

  <br>

  See `scripts/` for example usage