## KSU Compression Algorithm Implementation ##

Algortihm 1 from [Nearest-Neighbor Sample Compression: Efficiency, Consistency, Infinite Dimensions](https://arxiv.org/abs/1705.08184)

Installation
------------
* With pip: `pip install ksu`
* From source:
    * `git clone --depth=1 https://github.com/nimroha/ksu_classifier.git`
    * `cd ksu_classifier`
    * `python setup.py install`
    
Usage
 -----

 ##### Command Line #####

This package provides two command line tools: `e-net` and `ksu`:

* `e-net` constructs an [epsilon net](https://en.wikipedia.org/wiki/Delone_set) for a given epsilon
* `ksu` runs the full algorithm

Both provide the -h flag to specify the arguments, and both can save the result to the disk in [numpy's .npz](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) format

 <br>

 ##### Code #####

 This package provides a class `KSU(Xs, Ys, metric, [gram, prune, logLevel, n_jobs])`
 
 `Xs` and `Ys` are the data points and their respective labels as [numpy  arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html) 
 
 `metric` is either a callable to compute the metric or a string that names one of our provided metrics (print `KSU.METRICS.keys()` for the full list)
 
 `gram` _(optional, default=None)_ a precomputed [gramian matrix](http://mathworld.wolfram.com/GramMatrix.html), will be calculated if not provided.
 
 `prune` _(optional, default=False)_ a boolean indicating whether to prune the compressed set or not (Algorithm 2 from [Near-optimal sample compression for nearest neighbors](https://arxiv.org/abs/1404.3368))

 `logLevel` _(optional, default='CRITICAL')_ a string indicating the logging level (set to 'INFO' or 'DEBUG' to get more information)

 `n_jobs` _(optional, default=1)_ an integer defining how many cpus to use (scipy logic), pass -1 to use all. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
 
  <br>
 
  `KSU` provides a method `compressData([delta, minCompress, maxCompress, greedy, stride, logLevel, numProcs])`

  Which selects the subset with the lowest estimated error with confidence `1 - delta`. Can take arguments:

  `delta` _(optional, default=0.1)_ confidence for error upper bound

  `minCompress` _(optional, default=0.05)_ minimal compression ratio

  `maxCompress` _(optional, default=0.1)_ maximum compression ratio

  `greedy` _(optional, default=True)_ whether to use greedy or hierarichal strategy for net construction

  `stride` _(optional, default=200)_ how many gammas to skip between each iteration (since similar gammas will produce similar nets)

  `logLevel` _(optional, default='CRITICAL')_ a string indicating the logging level (set to 'INFO' or 'DEBUG' to get more information)

  `numProcs` _(optional, default=1)_ number of processes to use

  <br>
  
  You can then run `getClassifier()` which returns a 1-NN Classifer (based on [sklearn's K-NN](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)) fitted to the compressed data.
  
  Or, run `getCompressedSet()` to get the compressed data as a tuple of numpy arrays `(compressedXs, compressedYs)`.

  <br>

  See `scripts/` for example usage


  ##### Built-in metrics #####

  ['chebyshev', 'yule', 'sokalmichener', 'canberra', 'EarthMover', 'rogerstanimoto', 'matching', 'dice', 'EditDistance', 'braycurtis', 'russellrao', 'cosine', 'cityblock', 'l1', 'manhattan', 'sqeuclidean', 'jaccard', 'seuclidean', 'sokalsneath', 'kulsinski', 'minkowski', 'mahalanobis', 'euclidean', 'l2', 'hamming', 'correlation', 'wminkowski']