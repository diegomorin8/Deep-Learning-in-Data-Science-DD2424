import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os

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
                print self.book_chars

                # Dimensions
                self.K = len(self.book_chars)
                self.book_size = len(self.book_data)

                # Mappings from char to ind and vice versa, mapped in a dictionary
                self.char_to_ind = { character:index for index, character in enumerate(self.book_chars)}
                self.ind_to_char = { index:character for index,character in enumerate(self.book_chars)}
            
class RNN: 
        # Constructor
        def __init__(self, K ,m = 100, eta = 0.05, seq_length = 25):
                
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
                
                # Gradient of U weights
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
        
        def char_to_hot(self, X_in, Y_in, length = 25):
            
            # Initialize one hot matrices
            X = np.zeros((self.book.K,self.RNN.seq_length))
            Y = np.zeros((self.book.K,self.RNN.seq_length))

            # One hot encoding
            for i in range(length):
                X[self.book.char_to_ind[X_in[i]],i] = 1
                Y[self.book.char_to_ind[Y_in[i]],i] = 1
             
            return X,Y
        
        def forward_pass(self, X, Y, prev_hidden):
            
            # Reinitialize hidden states
            self.RNN.hidden[:,:] = 0
            # The first state is set as the last step from the previous iteration
            self.RNN.hidden[:,0] = prev_hidden

            # Probabilities matrix initialization
            p = np.zeros((self.book.K,self.RNN.seq_length))

            # Cost initialization
            cost = 0

            for t in range(self.RNN.seq_length): 

                # Get the new hidden state
                a_t = np.dot(self.RNN.W, self.RNN.hidden[:,t]) + np.dot(self.RNN.U, X[:,t]) + self.RNN.b.squeeze()
                
                self.RNN.hidden[:,t + 1] = np.tanh(a_t)
                
                # Get the probabilities
                o_t = np.dot(self.RNN.V, self.RNN.hidden[:,t + 1]) + self.RNN.c.squeeze()
               
                e = np.exp(o_t)
                
                # Softmax
                if t == 0: 
                    p[:,t] = e / e.sum()
                else: 
                    p[:,t] = e / e.sum()

                cost += -np.log( np.sum(p[:,t]*Y[:,t]))
            
            # Save the last hidden state
            prev_hidden = self.RNN.hidden[:,-1]                 
            return p, cost, prev_hidden
        
        def backward_pass(self, X, Y, p):
            
            # Initialization
            dh_next = np.zeros((self.RNN.m,1))
            # Reinitialize the gradients.
            self.grad_to_zero()
                        
            # Backward-pass
            for t in reversed(range(self.RNN.seq_length)):

                    # Gradient
                    g = p[:,t] - Y[:,t]
                    
                    # Bias c gradient update
                    self.RNN.grad_c[:,0] += g
                    
                    # Weights V gradient update
                    self.RNN.grad_V += np.outer(g, self.RNN.hidden[:,t+1])
                    
                    g = g.reshape(g.shape[0],1)
                    
                    # Hidden layer gradient
                    first_component = (1 - self.RNN.hidden[:,t+1] ** 2).reshape((1 - self.RNN.hidden[:,t+1] ** 2).shape[0],1)
                    dh = first_component * (np.dot(self.RNN.V.T, g) + dh_next)

                    # Weights U gradient
                    self.RNN.grad_U += np.outer(dh, X[:,t])
                    
                    # Weights W gradient
                    self.RNN.grad_W += np.outer(dh, self.RNN.hidden[:,t])
                    
                    # Bias b gradient
                    self.RNN.grad_b[:,0] += dh[:,0]
                    
                    # Next character hidden gradient
                    dh_next = np.dot(self.RNN.W.T, dh)
                    
            # This function ensures that the gradients does not grow above or below [-5,5]
            self.RNN.grad_U = np.clip(self.RNN.grad_U, -5, 5)
            self.RNN.grad_W = np.clip(self.RNN.grad_W, -5, 5)
            self.RNN.grad_V = np.clip(self.RNN.grad_V, -5, 5)
            self.RNN.grad_b = np.clip(self.RNN.grad_b, -5, 5)
            self.RNN.grad_c = np.clip(self.RNN.grad_c, -5, 5)

        def update(self):
            
            # Update W
            self.RNN.Agrad_W += self.RNN.grad_W * self.RNN.grad_W
            self.RNN.W += - self.RNN.eta * self.RNN.grad_W / np.sqrt(self.RNN.Agrad_W +  1e-8)

            # Update U
            self.RNN.Agrad_U += self.RNN.grad_U * self.RNN.grad_U
            self.RNN.U += - self.RNN.eta * self.RNN.grad_U / np.sqrt(self.RNN.Agrad_U + 1e-8)

            # Update V
            self.RNN.Agrad_V += self.RNN.grad_V * self.RNN.grad_V
            self.RNN.V += - self.RNN.eta * self.RNN.grad_V / np.sqrt(self.RNN.Agrad_V + 1e-8)
            

            # Update b
            self.RNN.Agrad_b += self.RNN.grad_b * self.RNN.grad_b
            self.RNN.b += - self.RNN.eta * self.RNN.grad_b / np.sqrt(self.RNN.Agrad_b +  1e-8)

            # Update c
            self.RNN.Agrad_c += self.RNN.grad_c * self.RNN.grad_c
            self.RNN.c += - self.RNN.eta * self.RNN.grad_c / np.sqrt(self.RNN.Agrad_c +  1e-8)
            
        def train(self, epochs = 10):
            
            # Iterations counter
            Iteration = 0

            # Text files to save parameters
            if not os.path.exists("results"):
                os.makedirs("results")
            if not os.path.exists('results/text.txt'):
                open('results/text.txt', 'a').close()
            if not os.path.exists('results/loss.txt'):
                open('results/loss.txt', 'a').close() 
            
            txt = open("results/text.txt", 'w')
            loss = open("results/loss.txt", 'w')
            
            # Length of synthsized text
            n = 200
            
            # Total loss
            self.total_loss = []
                  
            # First text sythesizing
            text = self.Synthesize_text(self.book.char_to_ind['0'], n)
            text_char = ''.join([self.book.ind_to_char[character] for character in text])
            txt.write('Epoch : ' + str(0) + ' - Iteration : ' + str(0) + 'Synthesized text : \n' + text_char + '\n')
            print 'Epoch : ' + str(0) + ' - Iteration : ' + str(0) + 'Synthesized text : \n' + text_char + '\n'
            
            for i in range(epochs): 
                
                print 'Epoch : ' + str(i)
                # Position of the test
                e = 0
                # Previous state. 
                prev_state = self.RNN.hidden_init[:,0]
                
                # Character's sequences iteration
                while e < (self.book.book_size - self.RNN.seq_length - 1): 
                    
                    # Increase the counter
                    Iteration += 1
                    
                    # Get the text of length 25
                    X_chars = self.book.book_data[e:(e + self.RNN.seq_length)]
                    Y_chars = self.book.book_data[e + 1:(e + self.RNN.seq_length + 1)]
		   
                    # Text sintesizing every 10000 iterations
                    if Iteration % 10000 == 0:
                        text = self.Synthesize_text(self.book.char_to_ind[X_chars[0]], n)
                        text_char = ''.join([self.book.ind_to_char[character] for character in text])
                        txt.write('\nEpoch : ' + str(i) + ' - Iteration : ' + str(Iteration) + ' - Cost: ' + str(smooth_loss) + ' - Synthesized text : \n' + text_char + '\n\n') 
                    # Print the text
                    if Iteration % 10000 == 0: 
                        print 'Epoch : ' + str(i) + ' - Iteration : ' + str(Iteration) + 'Synthesized text : \n' + text_char + '\n'
                    
                                       
                    # One hot encoding
                    X,Y = self.char_to_hot(X_chars, Y_chars)
                    
                    # Forward - pass
                    p, cost, prev_state = self.forward_pass(X, Y, prev_state)
                    
                    # Cost smoothing
                    if Iteration == 1:
                        smooth_loss = cost
                    else: 
                        smooth_loss = 0.999*smooth_loss + 0.001*cost;
                    
                    # Save the loss every iteration 
                    self.total_loss.append(smooth_loss)
                    
                    # Save loss in a text file every 1000 iterations
                    if Iteration % 1000 == 0:
                        loss.write('Loss : ' + str(smooth_loss) + ' Iteration : ' + str(Iteration) + ' Epoch : ' + str(i) + '\n' )
                        print 'Loss : ' + str(smooth_loss) + ' Iteration : ' + str(Iteration) + ' Epoch : ' + str(i)
                    
                    # Backward - pass
                    self.backward_pass(X, Y, p)
                    
                    # AdaGrad Update 
                    self.update()
                    
                    # Update e
                    e += self.RNN.seq_length
                    
                # The final characters of the book
               
                # Test strings
                X_chars = self.book.book_data[e:-2]
                Y_chars = self.book.book_data[e + 1:-1]   

                # Onte hot encoding
                X,Y = self.char_to_hot(X_chars, Y_chars, (self.RNN.seq_length - e))

                # Forward - pass
                p, cost, prev_state = self.forward_pass(X,Y,prev_state)

                # Backward - pass
                self.backward_pass(X, Y, p)

                # AdaGrad Update 
                self.update
            
            txt.close()
            loss.close()               
                
                

                
         
        def grad_check(self, weights, grads, X, Y, check = 10):
            
            # Initialize the previous state to 0
            prev_hidden = self.RNN.hidden_init[:,0]
            # Relative error to 0
            rel_error = np.zeros((check,1))
            # Delta used
            delta = 1e-4
            # To perform random checking
            rand_i = np.random.uniform(0, weights.shape[0] - 1, check).astype(int)
            rand_j = np.random.uniform(0, weights.shape[1] - 1, check).astype(int)
            rand_index = np.array([ rand_i, rand_j])
            it = 0
            # For each random pair
            for index in rand_index.T:
                # Cost for weights + delta
                weights[index[0],index[1]] = weights[index[0],index[1]] + delta
                _, cost0, _ = self.forward_pass(X, Y, prev_hidden)
                # Cost for weights - delta
                weights[index[0],index[1]] = weights[index[0],index[1]] - 2*delta
                _, cost1, _ = self.forward_pass(X, Y, prev_hidden)
                # Compute the gradient
                grad_numerical = (cost0 - cost1) / (2*delta)
                # Analytical gradients back to the orginal value
                weights[index[0],index[1]] = weights[index[0],index[1]] + delta
                # Gradients comparison
                rel_error[it] = abs(grads[index[0],index[1]] - grad_numerical) / abs(grad_numerical + grads[index[0],index[1]] + np.finfo(float).eps)
                if not math.isnan(rel_error[it]) and rel_error[it] > 1e-4:
                    print 'Error with: %f, %f => %e ' % (grad_numerical, grads[index[0],index[1]], rel_error[it])
                it = it + 1
            
            return rel_error

        def grad_to_zero(self):

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


def training(epochs = 10): 
    
    # Random seed to ensure reproducibility
    random.seed(200)
    
    # Load the book
    book = LoadBook()

    # Initialize the network
    rnn = RNN(book.K)

    # Initialize functions
    functions = Functions(rnn,book)
    
    # Training
    functions.train(epochs)
    
    # Save synthesize
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists('results/text.txt'):
        open('results/syn.txt', 'a').close()
           
    # Save synthesized text        
    syn = open("results/syn.txt", 'w')
    text = functions.Synthesize_text(functions.book.char_to_ind['H'], 1000)
    text_char = ''.join([functions.book.ind_to_char[character] for character in text])
    syn.write(text_char)   
    syn.close()
    
    # Plot the loss
    plt.figure()
    plt.plot(functions.total_loss)
    plt.show()
    
def overfitting(epochs = 100): 
    
    # Random seed to ensure reproducibility
    random.seed(200)
    
    # Load the book
    book = LoadBook('data/overfiting.txt')

    # Initialize the network
    rnn = RNN(book.K)

    # Initialize functions
    functions = Functions(rnn,book)
    
    # Training
    functions.train(epochs)
    
    # Save synthesize
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists('results/text.txt'):
        open('results/syn.txt', 'a').close()
           
    # Save synthesized text        
    syn = open("results/syn.txt", 'w')
    text = functions.Synthesize_text(functions.book.char_to_ind['H'], 1000)
    text_char = ''.join([functions.book.ind_to_char[character] for character in text])
    syn.write(text_char)   
    syn.close()
    
    # Plot the loss
    plt.figure()
    plt.plot(functions.total_loss)
    plt.show()
    
            
def gradient_test(check = 10,m = 5):
    
    # Random seed
    random.seed(200)
    
    # Load the book
    book = LoadBook()

    # Initialize the network
    rnn = RNN(book.K, 5)

    # Intialize the previous state
    prev_hidden = rnn.hidden_init[:,0]
    
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
    p, cost, prev_hidden = functions.forward_pass(X, Y, prev_hidden)

    # Gradients computation
    functions.backward_pass(X, Y, p)
    
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
    
    # plotting all the errors together
    plt.figure(figsize=(15, 10))
    plt.plot(error_V)
    plt.plot(error_W)
    plt.plot(error_U)
    plt.plot(error_b)
    plt.plot(error_c)
    plt.axis([0, 1000, 0, 8e-6])
    plt.legend(labels = ["error_V","error_W","error_U","error_b","error_c"])
    plt.show()


if __name__ == "__main__":
    #gradient_test(1000)
    #overfitting
    training()
    
