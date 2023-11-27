import numpy as np
import matplotlib.pyplot as plt
import time

def convert_csv(images,labels,outfile,n):# n is the number of images(60k for training and 10k for testing)
    image_file = open(images,"rb") # rb for reading as a binary file
    label_file = open(labels,"rb")
    csv_file = open(outfile,"w")#w for write mode

    image_file.read(16)# reads 16bytes
    label_file.read(8) # reads 8 bytes
    _images = []

    for i in range(n):
        img = [ord(label_file.read(1))]
        for j in range(784):
            img.append(ord(image_file.read(1)))
        _images.append(img)
    
    for i in _images:
        csv_file.write(",".join(str(pixels) for pixels in img)+ "\n")
    
    image_file.close()
    label_file.close()
    csv_file.close()

mnist_train_x = "/Users/RithishChanalu/Documents/mnist/train-images-idx3-ubyte"
mnist_train_y = "/Users/RithishChanalu/Documents/mnist/train-labels-idx1-ubyte"
mnist_test_x = "/Users/RithishChanalu/Documents/mnist/t10k-images-idx3-ubyte"
mnist_test_y = "/Users/RithishChanalu/Documents/mnist/t10k-labels-idx1-ubyte"

convert_csv(mnist_train_x,mnist_train_y,"/Users/RithishChanalu/Documents/mnist/train.csv",60000)
convert_csv(mnist_test_x,mnist_test_y,"/Users/RithishChanalu/Documents/mnist/test.csv",10000)

train_file = open("/Users/RithishChanalu/Documents/mnist/train.csv","r")
train_list = train_file.readlines()
train_file.close()
#print(len(train_list))

values = train_list[100].split(",") # first value in values is the label number
image_array = np.asfarray(values[1:]).reshape((28,28))
plt.imshow(image_array, cmap = "Greys", interpolation = "None")
plt.show()

test_file = open("/Users/RithishChanalu/Documents/mnist/test.csv","r")
test_list = test_file.readlines()
test_file.close()
print(len(test_list))

class NN():
    def __init__(self):

        self.nn_size = [784,128,64,10]
        self.num_epochs = 10
        self.lr = 0.001

        input_l = self.nn_size[0]
        hidden_1 = self.nn_size[1]
        hidden_2 = self.nn_size[2]
        output_l = self.nn_size[3]

        self.weights = {
        'w1':np.random.randn(hidden_1, input_l) * np.sqrt(1. / hidden_1), # 128*784
        'w2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2), # 64*128
        'w3':np.random.randn(output_l, hidden_2) * np.sqrt(1. / output_l) # 10*64
    }

        pass
    
    def forward_prop(self,x_train):
        weights = self.weights

        weights['a0'] = x_train

         # input layer to hidden layer 1
        weights['z1'] = np.dot(weights["w1"], weights['a0'])
        weights['a1'] = self.sigmoid(weights['z1'])

        # hidden layer 1 to hidden layer 2
        weights['z2'] = np.dot(weights["w2"], weights['a1'])
        weights['a2'] = self.sigmoid(weights['z2'])

        # hidden layer 2 to output layer
        weights['z3'] = np.dot(weights["w3"], weights['a2'])
        weights['a3'] = self.softmax(weights['z3'])

        return weights['a3']

    def backward_prop(self,y_train,output):

        weights = self.weights
        delta_w = {}

      # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(weights['z3'], derivative=True)
        delta_w['w3'] = np.outer(error, weights['a2'])

      # Calculate W2 update
        error = np.dot(weights['a3'].T, error) * self.sigmoid(weights['z2'], derivative=True)
        delta_w['w2'] = np.outer(error, weights['a1'])

      # Calculate W1 update
        error = np.dot(weights['w2'].T, error) * self.sigmoid(weights['z1'], derivative=True)
        delta_w['w1'] = np.outer(error, weights['a0'])

        return delta_w
    
    def sigmoid(self,x,derivative = False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))
    
    def softmax(self, x, derivative=False):
      # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def update(self,change_w):
        '''
          Update network parameters according to update rule from
          Stochastic Gradient Descent.

          θ = θ - η * ∇J(x, y), 
              theta θ:            a network parameter (e.g. a weight w)
              eta η:              the learning rate
              gradient ∇J(x, y):  the gradient of the objective function,
                                  i.e. the change for a specific theta θ
      '''
      
        for key, value in change_w.items():
            self.weights[key] -= self.lr * value

    def train(self,train_list, test_list, output_nodes):
        start_t = time.time()
        for iteration in range(self.num_epochs):

            np.random.shuffle(train_list)

            for x in train_list:
                all_values = x.split(',')
                # scale and shift the inputs
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                # create the target output values (all 0.01, except the desired label which is 0.99)
                targets = np.zeros(output_nodes) + 0.01 # soft 
                # all_values[0] is the target label for this record
                targets[int(all_values[0])] = 0.99
                output = self.forward_prop(inputs)
                changes_w = self.backward_prop(targets, output)
                self.update(changes_w)
          
            accuracy = self.accuracy(test_list, output_nodes)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
              iteration+1, time.time() - start_t, accuracy * 100
          ))
     
        pass

    def accuracy(self,test_list,output_nodes): # for accuracy computation
        '''
          This function does a forward pass of x, then checks if the indices
          of the maximum value in the output equals the indices in the label
          y. Then it sums over each prediction and calculates the accuracy.
      '''
        tests = []

        for x in test_list:
            all_values = x.split(',')
            # scale and shift the inputs
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = np.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[int(all_values[0])] = 0.99
            output = self.forward_prop(inputs)
            pred = np.argmax(output)
            tests.append(pred == np.argmax(targets))
      
        return np.mean(tests)





mnist = NN()
mnist.train(train_list, test_list, 10)
