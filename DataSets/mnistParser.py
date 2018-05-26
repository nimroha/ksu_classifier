from mnist import MNIST

mnist_dir = 'C:\Users\yuvalnissan\Desktop\MNIST'
mndata = MNIST(mnist_dir)
mndata.gz = True
mndata.load_training()
mndata.load_testing()

train = mndata.train_images
test = mndata.test_images