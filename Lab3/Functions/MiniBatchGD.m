function [Wstar, bstar, val_loss, train_loss] = MiniBatchGD(X, Y, X_val, Y_val, GDparams, W, b ,lambda, display, eta_decay, mW, mb, rho, X_test, y_test)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% where
%   - X: contains all the training images
%   - Y: the labels for the training images
%   - W, b: are the initial values for the network’s parameters
%   - lambda: is the regularization factor in the cost function 
%   - GDparams: is an object containing the parameter values n batch, eta
%               and n epochs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 11
    accuracy = 0;
    moments_check = 0; 
    eta_decay = 1;
elseif nargin < 12
    accuracy = 0;
    moments_check = 0;
elseif nargin < 13
    accuracy = 0;
    moments_check = 1;
    rho = 0.9;
elseif nargin < 14
    accuracy = 0;
    moments_check = 1; 
else 
    accuracy = 1; 
    moments_check = 1; 
end
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
        [P,h,s] = EvaluateClassifier( Xbatch, W, b);

        % Backward pass
        [Grad_W, Grad_b] = ComputeGradients( Xbatch, Ybatch, P, W, h, s, lambda );
        
        if moments_check == 0
            
            for i = 1:size(W,2)
                % Update network parameters as in the lecture notes
                W{i} = W{i} - eta*Grad_W{i};
                b{i} = b{i} - eta*Grad_b{i};
            end
            
        elseif moments_check == 1
            
            for i = 1:size(W,2)
                % Update network parameters as in the lecture notes
                mW{i} = rho*mW{i} + eta*Grad_W{i};
                mb{i} = rho*mb{i} + eta*Grad_b{i};

                W{i} = W{i} - mW{i};
                b{i} = b{i} - mb{i};
            end

        end
        
    end
    eta = eta_decay * eta;
    % Validation and trining loss
    train_loss(ep + 1) = ComputeCost( X, Y, W, b, lambda );
    val_loss(ep + 1) = ComputeCost( X_val, Y_val, W, b, lambda );
    if accuracy == 1
        acc = ComputeAccuracy( X_test, y_test, W, b);
    end
    if display == 2
        if accuracy == 0
           disp( sprintf(' Epoch: %f - Validation loss: %f - Training loss: %f', ep, val_loss(ep+1), train_loss(ep +1)))
        else
            disp( sprintf(' Epoch: %f - Validation loss: %f - Training loss: %f - Accuracy: %f', ep, val_loss(ep+1), train_loss(ep +1), acc))
        end
    end
    if train_loss(ep + 1) >= 3*train_loss(1)
        disp( ' Bad paramters ');
        break;
    elseif isnan(train_loss(ep+1))
        disp( ' Bad paramters ');
        break;
    end
        

end

% We output the trained parameters
Wstar = W;
bstar = b;

end