import numpy as np
import matplotlib.pyplot as plt
import time

def convert_csv(images, labels, outfile, n):
    image_file = open(images, "rb")
    label_file = open(labels, "rb")
    csv_file = open(outfile, "w")

    image_file.read(16)
    label_file.read(8)
    _images = []

    for i in range(n):
        img = [ord(label_file.read(1))]
        for j in range(784):
            img.append(ord(image_file.read(1)))
        _images.append(img)

    for i in _images:
        csv_file.write(",".join(str(pixels) for pixels in i) + "\n")

    image_file.close()
    label_file.close()
    csv_file.close()

mnist_train_x = "/Users/RithishChanalu/Documents/mnist/train-images-idx3-ubyte"
mnist_train_y = "/Users/RithishChanalu/Documents/mnist/train-labels-idx1-ubyte"
mnist_test_x = "/Users/RithishChanalu/Documents/mnist/t10k-images-idx3-ubyte"
mnist_test_y = "/Users/RithishChanalu/Documents/mnist/t10k-labels-idx1-ubyte"

convert_csv(mnist_train_x, mnist_train_y, "/Users/RithishChanalu/Documents/mnist/train.csv", 60000)
convert_csv(mnist_test_x, mnist_test_y, "/Users/RithishChanalu/Documents/mnist/test.csv", 10000)

train_file = open("/Users/RithishChanalu/Documents/mnist/train.csv", "r")
train_list = train_file.readlines()
train_file.close()

test_file = open("/Users/RithishChanalu/Documents/mnist/test.csv", "r")
test_list = test_file.readlines()
test_file.close()

class NN():
    def __init__(self):
        
        self.nn_size = [784, 128, 64, 10]
        self.num_epochs = 20
        self.lr = 0.001

        input_l = self.nn_size[0]
        hidden_1 = self.nn_size[1]
        hidden_2 = self.nn_size[2]
        output_l = self.nn_size[3]

        self.weights = {
            'w1': np.random.randn(hidden_1, input_l) * np.sqrt(1. / hidden_1),
            'w2': np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'w3': np.random.randn(output_l, hidden_2) * np.sqrt(1. / output_l)
        }

    def forward_prop(self, x_train):
        
        weights = self.weights
        weights['a0'] = x_train

        weights['z1'] = np.dot(weights["w1"], weights['a0'])
        weights['a1'] = self.sigmoid(weights['z1'])

        weights['z2'] = np.dot(weights["w2"], weights['a1'])
        weights['a2'] = self.sigmoid(weights['z2'])

        weights['z3'] = np.dot(weights["w3"], weights['a2'])
        weights['a3'] = self.softmax(weights['z3'])

        return weights['a3']

    def backward_prop(self, y_train, output):
        
        weights = self.weights
        delta_w = {}

        cost = 2 * (output - y_train) / output.shape[0] * self.softmax(weights['z3'], derivative=True)
        delta_w['w3'] = np.outer(cost, weights['a2'])

        cost = np.dot(weights['w3'].T, cost) * self.sigmoid(weights['z2'], derivative=True)
        delta_w['w2'] = np.outer(cost, weights['a1'])

        cost = np.dot(weights['w2'].T, cost) * self.sigmoid(weights['z1'], derivative=True)
        delta_w['w1'] = np.outer(cost, weights['a0'])

        return delta_w

    def sigmoid(self, x, derivative=False):

        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
        return 1 / (1 + np.exp(-x))

    def softmax(self, x, derivative=False):

        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def update(self, delta_w):

        for key, value in delta_w.items():
            self.weights[key] -= self.lr * value

    def train(self, train_list, test_list, output_nodes):
        
        start_t = time.time()
        for iteration in range(self.num_epochs):
            np.random.shuffle(train_list)
            for x in train_list:
                all_values = x.split(',')
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                targets = np.zeros(output_nodes) + 0.01
                targets[int(all_values[0])] = 0.99
                output = self.forward_prop(inputs)
                delta_w = self.backward_prop(targets, output)
                self.update(delta_w)
                loss = np.mean((output - targets) ** 2) 
                epoch_loss += loss

            epoch_loss /= len(train_list)
            losses.append(epoch_loss)

            accuracy = self.accuracy(test_list, output_nodes)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration + 1, time.time() - start_t, accuracy * 100
            ))
        self.plot_loss(losses)

    def accuracy(self, test_list, output_nodes):
        
        tests = []
        for x in test_list:
            all_values = x.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            output = self.forward_prop(inputs)
            prediction = np.argmax(output)
            tests.append(prediction == np.argmax(targets))

        return np.mean(tests)

    def plot_loss(self, losses):
        plt.plot(range(1, len(losses) + 1), losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Function During Training')
        plt.show()


mnist = NN()
mnist.train(train_list, test_list, 10)
