import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

class XorNN():
    def __init__(self):
        self.inodes = 2
        self.hnodes = 2
        self.onodes = 1
        self.lr = 0.1
        self.wih = np.random.rand(self.hnodes,self.inodes)-0.5
        self.woh = np.random.rand(self.onodes,self.hnodes)-0.5
        self.activation_fn = lambda x : sp.expit(x)
        self.inputs = np.array([[0,0,1,1],[0,1,0,1]])
        self.outputs = np.array([0,1,1,0])
        self.loss = []

#forward propogation
    def forward_prop(self,i):
            self.input_hidden = np.dot(self.wih,self.inputs[:,i])
            self.output_hidden = self.activation_fn(self.input_hidden)
            input_out = np.dot(self.woh,self.output_hidden)
            self.output_out = self.activation_fn(input_out)
            

    def backward_prop(self,i):
        self.e_out = self.outputs[i] - self.output_out
        self.e_hidden = np.dot(np.transpose(self.woh),self.e_out)
        
    def update(self):
        # Update weights for the output layer
        lr = 0.1
        e_out = self.e_out
        self.woh = self.woh + (lr * np.outer((e_out * self.output_out * (1 - self.output_out)),np.transpose(self.output_hidden)))
        print(self.woh)
        # Update weights for the hidden layer
        e_hidden = self.e_hidden

        self.wih = self.wih + (lr * np.outer((e_hidden * self.output_hidden * (1 - self.output_hidden)),np.transpose(self.input_hidden)))
        print(self.wih)
     
        
    
    def train(self):
        num_epochs = 10000
        for epoch in range(num_epochs):
            for i in range(4):
                self.forward_prop(i)
                self.backward_prop(i)
                self.update()
                loss = np.mean(0.5 * (self.outputs[i] - self.output_out)**2)
                self.loss.append(loss)

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MSE loss function graph')
        plt.show()

    def test(self,input):
        input_hidden = np.dot(self.wih,np.array(input))
        output_hidden = self.activation_fn(input_hidden)
        input_out = np.dot(self.woh,output_hidden)
        output_out = self.activation_fn(input_out)
        print(output_out)

xor = XorNN()
xor.train()
xor.plot_loss()
input = [0,1]
xor.test(input)



    







        
        


