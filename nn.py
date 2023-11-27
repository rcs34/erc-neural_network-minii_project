import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

class XorNN():
    def __init__(self):
        self.inodes = 2
        self.hnodes = 2
        self.onodes = 1
        self.lr = 0.1
        self.wih = np.random.rand(self.hnodes,self.inodes) - 0.005
        self.woh = np.random.rand(self.onodes,self.hnodes) - 0.005
        self.activation_fn = lambda x : sp.expit(x)
        self.inputs = np.array([[0,0,1,1],[0,1,0,1]])
        self.outputs = np.array([0,1,1,0])
        self.loss = []
        self.t_ex = self.inputs.shape[1]

#forward propogation
    def forward_prop(self):
            self.input_hidden = np.dot(self.wih,self.inputs)
            self.output_hidden = self.activation_fn(self.input_hidden)
            input_out = np.dot(self.woh,self.output_hidden)
            self.output_out = self.activation_fn(input_out)
            
            

    def backward_prop(self):
        self.e_out = self.outputs - self.output_out
        self.delta_woh = np.reshape(np.dot(self.e_out,self.input_hidden.T)/self.t_ex, self.woh.shape)
        #adjusted_woh = self.woh[:, :self.t_ex]
        self.e_hidden = np.dot(self.woh.T,self.e_out) * self.input_hidden * (1-self.input_hidden)
        self.delta_wih = np.reshape(np.dot(self.e_hidden, self.inputs.T)/self.t_ex, self.wih.shape)
        
    def update(self):
        lr = 0.1
        self.woh = self.woh - (lr * self.delta_woh)
        print(self.woh)
        
        self.wih = self.wih - (lr * self.delta_wih)
        print(self.wih)
     
        
    
    def train(self):
        num_epochs = 10000
        for epoch in range(num_epochs):
            self.forward_prop()
            loss = -(1/self.t_ex)*np.sum(self.outputs*np.log(self.output_out)+(1-self.outputs)*np.log(1-self.output_out))
            self.loss.append(loss)
            self.backward_prop()
            self.update()

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss function graph')
        plt.show()

    def test(self,input):
        input_hidden = np.dot(self.wih,np.array(input))
        output_hidden = self.activation_fn(input_hidden)
        input_out = np.dot(self.woh,output_hidden)
        output_out = self.activation_fn(input_out)
        print(output_out)

xor = XorNN()
print(xor.t_ex)
xor.train()
xor.plot_loss()
input = [1,1]
xor.test(input)

