function [Wstar, bstar, val_loss, train_loss] = MiniBatchGD(X, Y, X_val, Y_val, GDparams, W, b, lambda, display)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% where
%   - X: contains all the training images
%   - Y: the labels for the training images
%   - W, b: are the initial values for the network’s parameters
%   - lambda: is the regularization factor in the cost function 
%   - GDparams: is an object containing the parameter values n batch, eta
%               and n epochs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Obtain training parameters

% Number of images in the data set
N = size(X,2);

% Size of mini batches
n_batch = GDparams.n_batch;

% Learning rate
eta = GDparams.eta;

% Number of epochs
n_epochs = GDparams.n_epochs;

% Training set initialization 
train_loss = zeros(n_epochs+1,1);
train_loss(1) = ComputeCost( X, Y, W, b, lambda );

% Training set initialization 
val_loss = zeros(n_epochs+1,1);
val_loss(1) = ComputeCost( X_val, Y_val, W, b, lambda );
    
for ep=1:n_epochs
    % Prepare each batch. Code given in the lecture notes
    for j=1:N/n_batch
        
        % Mini batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, inds);
        Ybatch = Y(:, inds);
        
        % Forward pass
        P = EvaluateClassifier( Xbatch, W, b );

        % Backward pass
        [gradW, gradb] = ComputeGradients( Xbatch, Ybatch, P, W, lambda );

        % Update network parameters as in the lecture notes
        W = W - eta*gradW;
        b = b - eta*gradb;
        
    end
    
    % Validation and trining loss
    train_loss(ep + 1) = ComputeCost( X, Y, W, b, lambda );
    val_loss(ep + 1) = ComputeCost( X_val, Y_val, W, b, lambda );
    if display == 2
        disp( sprintf(' Epoch: %f - Validation loss: %f - Training loss: %f', ep, val_loss, train_loss))
    end

end

% We output the trained parameters
Wstar = W;
bstar = b;

end