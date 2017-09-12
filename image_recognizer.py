from sklearn import datasets
import pylab as pl
def main():
    digits = datasets.load_digits()
    print(digits.images.shape)
    pl.imshow(digits.images[0],cmap = None)
if __name__ == '__main__':
    main()