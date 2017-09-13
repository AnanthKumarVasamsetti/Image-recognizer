from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def calculate_sigmoid(z):
    return (1/(1+np.exp(-z)))

def calculate_z(weights, features, labels):
    intercept = 0
    z = np.dot(weights, features) + intercept
    sigmoid_matrix = calculate_sigmoid(z)
    print(sigmoid_matrix)
def main():
    digits = datasets.load_digits()
    features = np.array(digits.data, 'int16')
    labels = np.array(digits.target,'int')
    features = features.T
    labels = labels.T
    weights = np.ones((1,features.shape[0]))
    calculate_z(weights, features, labels)
if __name__ == '__main__':
    main()