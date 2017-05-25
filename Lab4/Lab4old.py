import numpy as np
import matplotlib.pyplot as plt
import math
import random
from random import uniform

# Load the book and get the main characteristics from the book
class LoadBook:
    # Constructor
        def __init__(self, book_fname  = 'data/goblet.txt'):
            
                # Load the book
                self.book_data = open(book_fname,'r').read()

                # Get the unique characters as a list
                self.book_chars = ''.join(set(self.book_data))
                self.book_chars = list(sorted(self.book_chars))

                # Dimensions
                self.K = len(self.book_chars)
                self.book_size = len(self.book_data)

                # Mappings from char to ind and vice versa, mapped in a dictionary
                self.char_to_ind = { character:index for index, character in enumerate(self.book_chars)}
                self.ind_to_char = { index:character for index,character in enumerate(self.book_chars)}
            
class RNN: 
        # Constructor
        def __init__(self, K ,m = 100, eta = 0.1, seq_length = 25):
                
                # Size of the hidden layer
                self.m = m

                # Learning rate
                self.eta = eta

                # Batch size 
                self.seq_length = seq_length
                
                # Initizalize hidden
                self.hidden = np.zeros((m,seq_length + 1))
                
                # The hidden layer is updated during training, so: 
                self.hidden_init = np.zeros((m,1))

                # Bias vector of the hidden layer
                self.b = np.zeros((m,1))

                # Bias vector of the outputs
                self.c = np.zeros((K,1))

                # Sigma
                sig = 0.01

                # Weight matrix of the input
                self.U = np.random.randn(m,K)*sig
                #self.U = np.ones((m,K))*0.1
                
                # Weight matrix of the reccurrent conection 
                self.W = np.random.randn(m,m)*sig
                #self.W = np.ones((m,m))*0.1
                
                # Weight matrix of the output
                self.V = np.random.randn(K,m)*sig
                #self.V = np.ones((K,m))*0.1
                
                # Gradient of b bias
                self.grad_b = np.zeros((m,1))
                
                # Gradient of c bias
                self.grad_c = np.zeros((K,1))

                # Gradient of V weights
                self.grad_V = np.zeros((K,m))
                
                # Gradient of W weigths
                self.grad_W = np.zeros((m,m))
                
                #Graident of U weights
                self.grad_U = np.zeros((m,K))
                
                # AdaGradient of b bias
                self.Agrad_b = np.zeros((m,1))
                
                # AdaGradient of c bias
                self.Agrad_c = np.zeros((K,1))

                # AdaGradient of V weights
                self.Agrad_V = np.zeros((K,m))
                
                # AdaGradient of W weigths
                self.Agrad_W = np.zeros((m,m))
                
                # AdaGraident of U weights
                self.Agrad_U = np.zeros((m,K))
                

                
class Functions: 
        # Constructor
        def __init__(self, RNN, book):
                
            # Save the RNN 
            self.RNN = RNN
            
            # Save book
            self.book = book 
        
        def Synthesize_text(self, x0, length):
            
            # States initialization 
            x_t = np.zeros((self.book.K,1))
            
            # First dummy charachter
            x_t[x0] = 1

            # We don't want to modify the value of the hidden in the network, just use it
            hidden = self.RNN.hidden_init

            # We need to store the text
            text = [] 

            for t in range (length):
                # Prediction
                a_t = np.dot(self.RNN.W,hidden) + np.dot(self.RNN.U,x_t) + self.RNN.b
                hidden = np.tanh(a_t)
                o_t = np.dot(self.RNN.V,hidden) + self.RNN.c
                e = np.exp(o_t)
                p_t = e / e.sum()

                # Randomly pick a character
                cp = np.cumsum(p_t)
                # Random number from normal distribution
                a = np.random.uniform()
                ind = np.array(np.nonzero(cp - a > 0))
                ind = ind[0,0]

                # take sampled index as an input to next sampling
                x_t = np.zeros((self.book.K,1))
                x_t[ind] = 1

                # Save the computed character
                text.append(ind)

            return text
        
        def char_to_hot(self, X_in, Y_in):
            
            # Initialize one hot matrices
            X = np.zeros((self.book.K,self.RNN.seq_length))
            Y = np.zeros((self.book.K,self.RNN.seq_length))

            # One hot encoding
            for i in range(self.RNN.seq_length):
                X[self.book.char_to_ind[X_in[i]],i] = 1
                Y[self.book.char_to_ind[Y_in[i]],i] = 1
             
            return X,Y
        
        def forward_pass(self, X, Y):
            
            # We don't want to modify the value of the hidden in the network, just use it
            self.RNN.hidden[:,-1] = self.RNN.hidden_init[:,0]

            # Probabilities matrix
            p = np.zeros((self.book.K,self.RNN.seq_length))

            # Cost initialization
            cost = 0

            for t in range(self.RNN.seq_length): 

                # find new hidden state
                a_t = np.dot(self.RNN.W, self.RNN.hidden[:,t]) + np.dot(self.RNN.U, X[:,t]) + self.RNN.b.squeeze()

                self.RNN.hidden[:,t + 1] = np.tanh(a_t)

                # unnormalized log probabilities for next chars o_t
                o_t = np.dot(self.RNN.V, self.RNN.hidden[:,t + 1]) + self.RNN.c.squeeze()

                e = np.exp(o_t)
                # Softmax
                if t == 0: 
                    p[:,t] = e / e.sum()
                else: 
                    p[:,t] = e / e.sum()

                cost += -np.log( np.sum(p[:,t]*Y[:,t]))
                       
            return p, cost
        
        def backward_pass(self, X, Y, p, cost):
            
            self.grads_to_zero()

            dh_next = self.RNN.hidden_init

            for t in reversed(range(self.RNN.seq_length)):

                    # gradient w.r.t. o_t
                    g = p[:,t] - Y[:,t]
                    
                    # Bias c gradient update
                    self.RNN.grad_c[:,0] += g
                    
                    # gradient w.r.t. V and c
                    self.RNN.grad_V += np.outer(g, self.RNN.hidden[:,t+1])
                    
                    g = g.reshape(g.shape[0],1)
                    
                    # gradient w.r.t. h, tanh nonlinearity
                    first_component = (1 - self.RNN.hidden[:,t+1] ** 2).reshape((1 - self.RNN.hidden[:,t+1] ** 2).shape[0],1)
                    dh = first_component * (np.dot(self.RNN.V.T, g) + dh_next)

                    # gradient w.r.t. U
                    self.RNN.grad_U += np.outer(dh, X[:,t])
                    
                    # gradient w.r.t W
                    self.RNN.grad_W += np.outer(dh, self.RNN.hidden[:,t])
                    
                    # gradient w.r.t. b
                    self.RNN.grad_b[:,0] += dh[:,0]
                    
                    # Next (previous) dh
                    dh_next = np.dot(self.RNN.W.T, dh)
                    
            # This function ensures that the gradients does not grow above or below [-5,5]
            #self.RNN.grad_U = np.clip(self.RNN.grad_U, -5, 5)
            #self.RNN.grad_W = np.clip(self.RNN.grad_W, -5, 5)
            #self.RNN.grad_V = np.clip(self.RNN.grad_V, -5, 5)
            #self.RNN.grad_b = np.clip(self.RNN.grad_b, -5, 5)
            #self.RNN.grad_c = np.clip(self.RNN.grad_c, -5, 5)

        def update(self):
            
            # Update W
            self.RNN.Agrad_W += self.RNN.grad_W * self.RNN.grad_W
            self.RNN.W += - self.RNN.eta * self.RNN.grad_W / np.sqrt(self.RNN.Agrad_W + np.finfo(float).eps)

            # Update U
            self.RNN.Agrad_U += self.RNN.grad_U * self.RNN.grad_U
            self.RNN.U += - self.RNN.eta * self.RNN.grad_U / np.sqrt(self.RNN.Agrad_U + np.finfo(float).eps)

            # Update V
            self.RNN.Agrad_V += self.RNN.grad_V * self.RNN.grad_V
            self.RNN.V += - self.RNN.eta * self.RNN.grad_V / np.sqrt(self.RNN.Agrad_V + np.finfo(float).eps)

            # Update b
            self.RNN.Agrad_b += self.RNN.grad_b * self.RNN.grad_b
            self.RNN.b += - self.RNN.eta * self.RNN.grad_b / np.sqrt(self.RNN.Agrad_b + np.finfo(float).eps)

            # Update c
            self.RNN.Agrad_c += self.RNN.grad_c * self.RNN.grad_c
            self.RNN.c += - self.RNN.eta * self.RNN.grad_c / np.sqrt(self.RNN.Agrad_c+ np.finfo(float).eps)
            
        def grads_to_zero(self):
            
            # Gradient of b bias
            self.RNN.grad_b = np.zeros((self.RNN.m,1))
                
            # Gradient of c bias
            self.RNN.grad_c = np.zeros((self.book.K,1))

            # Gradient of V weights
            self.RNN.grad_V = np.zeros((self.book.K,self.RNN.m))

            # Gradient of W weigths
            self.RNN.grad_W = np.zeros((self.RNN.m,self.RNN.m))

            #Graident of U weights
            self.RNN.grad_U = np.zeros((self.RNN.m,self.book.K))
         
        def grad_check(self, weights, grads, X, Y, check = 10):
        
            rel_error = np.zeros((check,1))
            delta = 1e-4
            rand_i = np.random.uniform(0, weights.shape[0] - 1, check).astype(int)
            rand_j = np.random.uniform(0, weights.shape[1] - 1, check).astype(int)
            rand_index = np.array([ rand_i, rand_j])
            it = 0
            for index in rand_index.T:
                weights[index[0],index[1]] = weights[index[0],index[1]] + delta
                _, cost0 = self.forward_pass(X, Y)
                weights[index[0],index[1]] = weights[index[0],index[1]] - 2*delta
                _, cost1 = self.forward_pass(X, Y)
                grad_numerical = (cost0 - cost1) / (2*delta)
                weights[index[0],index[1]] = weights[index[0],index[1]] + delta
                rel_error[it] = abs(grads[index[0],index[1]] - grad_numerical) / abs(grad_numerical + grads[index[0],index[1]] + np.finfo(float).eps)
                if not math.isnan(rel_error[it]) and rel_error[it] > 1e-4:
                    print 'Error with: %f, %f => %e ' % (grad_numerical, grads[index[0],index[1]], rel_error[it])
                it = it + 1
            
            return rel_error

            
            
def gradient_test(check = 10,m = 5):
    random.seed(200)
    # Load the book
    book = LoadBook()

    # Initialize the network
    rnn = RNN(book.K, 5)

    # Initialize functions
    functions = Functions(rnn,book)

    # Test strings
    X_chars = book.book_data[0:rnn.seq_length]
    Y_chars = book.book_data[1:rnn.seq_length+1]
 
    # Get the randomly Synthesized text
    text = functions.Synthesize_text(book.char_to_ind['d'], 20)
    
    # Onte hot encoding
    X,Y = functions.char_to_hot(X_chars, Y_chars)

    # Forward - pass
    p, cost = functions.forward_pass(X,Y)

    functions.backward_pass(X, Y, p, cost)
    
    print "Checking V gradients . . . "
    error_V = functions.grad_check(rnn.V, rnn.grad_V, X, Y, check)
    print "Checking W gradients . . . "
    error_W = functions.grad_check(rnn.W, rnn.grad_W, X, Y, check)
    print "Checking U gradients . . . "
    error_U = functions.grad_check(rnn.U, rnn.grad_U, X, Y, check)
    print "Checking b gradients . . . "
    error_b = functions.grad_check(rnn.b, rnn.grad_b, X, Y, check)
    print "Checking c gradients . . . "
    error_c = functions.grad_check(rnn.c, rnn.grad_c, X, Y, check)
    
    return error_V,error_W,error_U,error_b,error_c
