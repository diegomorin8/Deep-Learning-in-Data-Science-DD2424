function ass1_main( n_batch, eta, n_epochs, lambda_in, rho, eta_decay, all_data, display )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is divided in 3 main parts. The first one test the adapted
% functions that will be used in the last sections to compute the
% Mini-Batch descent Algorithm. In the second part, the main parameters,
% eta and lambda are tuned. Is in the last section were this function is
% used to train a network. 
%
% Inputs 
%   - n_batch : sixe of each mini batch in the Gradient descent. 
%   - eta : parameter used to give more or less importance to the
%                     calculated gradient in each epoch.
%   - n_epochs : number of training epochs. 
%   - lambda_in : value of lambda used in the training
%   - rho : weights the momentum term
%   - eta_decay : sets the decay for the eta term
%   - all_data : 1 - All the batches are used for training
%                2 - Only the first batch is used for training
%   - display : if set to:
%           - 0: nothing will be displayed and the function will be used to
%                train a network
%           - 1: only the first part of the assignment will be run and
%                displayed (functions test)
%           - 2. only the second part of the assignment will be run and
%                displayed (parameter tuning)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Introduction
%
% In this assignment a two layer network with multiple outputs will be
% trained to classify images from the dataset CIFAR-10. The network will be
% trained using minibatch gradient descent applied to a cost function that
% computes the cross-entropy loss of the classifier applied to the labelled
% training data and an L2 regularization term on the weight matrix.
  

    % The first step is to prepare the workspace

    %Prepare the workspace
    close all;
    % clear all;
    clc;

    if display == 1
        
        % Load the data
        % 
        % We will use the images in data_batch_1.mat as the training data. However
        % this data has to be modified and adapted to our requirements. This is
        % done by the function LoadBatch. This function loads the data we need, and
        % outputs 3 parameters: 
        %   - X: contains the image pixel data, has size dxN, is of type double or
        %     single and has entries between 0 and 1. N is the number of images
        %     (10000) and d the dimensionality of each image (3072=32×32×3).
        %   - Y: is K×N (K= # of labels = 10) and contains the one-hot representation
        %     of the label for each image.
        %   - N: is a vector of length N containing the label for each image. A note
        %     of caution. CIFAR-10 encodes the labels as integers between 0-9 but
        %     Matlab indexes matrices and vectors starting at 1. Therefore it may be
        %     easier to encode the labels between 1-10.

        [X_batch, Y_batch, y_batch] = LoadBatch('data_batch_1.mat');
        [X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
        [X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

        % Pre - processing

        mean_ = mean(X_batch, 2);
        X_batch = X_batch - repmat(mean_, [1, size(X_batch, 2)]);
        X_val = X_val - repmat(mean_, [1, size(X_val, 2)]);
        X_test = X_test - repmat(mean_, [1, size(X_test, 2)]);


        % Gradients Check

        lambda = 0; 
        error_W = [];
        error_b = [];

        num_checks = 100; 
        X_train = X_batch(1:num_checks,:);
        Y_train = Y_batch;

        % Number of features 
        [DataDim, ~] = size(X_train);

        % Number of nodes in the hidden layer
        HIDDEN_NODES = 50;

        % Number of labels
        [NumLabels, ~] = size(Y_train);

        % Size of the data set used in Xavier's initialization
        SizeDataSet = size(X_train,2);

        % Weight and bias initialization
        [W, b, mW, mb] = network_init(DataDim, NumLabels, HIDDEN_NODES, SizeDataSet);

        %
        % As we are going to use different mini batches to check the performance,
        % we want a loop. 
        for i = 1:100
            disp(sprintf(' Iteration = %i', i))
            % Numerical batch
            [ngrad_W, ngrad_b] = ComputeGradsNumSlow(X_train(:,i), Y_train(:,i), W, b, lambda, 1e-5);

            % We need P
            [P, h, s] = EvaluateClassifier(X_train(:,i), W, b);
            % Analytical batch
            [grad_W, grad_b] = ComputeGradients(X_train(:,i), Y_train(:,i), P, W, h, s, lambda);

            % Calculate the error according to the lab notes
            error_W(i) = norm(grad_W{1} - ngrad_W{1})/(norm(grad_W{1}) + norm(ngrad_W{1}));
            error_b(i) = norm(grad_b{1} - ngrad_b{1})/(norm(grad_b{1}) + norm(ngrad_b{1}));

            % The error has to be small enough
            if error_W(i) >= 1e-8
                disp(' Error in W gradient calculation ')

            elseif error_b(i) >= 1e-8
                disp(' Error in b gradient calculation ')

            end
        end
        

    else

        % 2. Mini batch gradient descent algprithm
        %
        % The main objetive is to train the network to classified 10 objects from
        % the pictures in the data set CIFAR-10 using the Mini Batch Gradient
        % Descent Algorithm. The implementation is going to be simplified being the
        % only parameters that can be tuned the following ones: 
        %   - n_batch: the size of the mini-batches
        %   - eta: the learning rate
        %   - n_epochs the number of runs through the whole training set.

        % 2.1 Prepare workspace

        % Load the data again
        [X_batch, Y_batch, y_batch] = LoadBatch('data_batch_1.mat');
        [X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
        [X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

        % 2.2 Weight matrix initialization
        %
        % We are using the random seed number 400 so we can get the same results as
        % in the lecture notes
        rng(400)

        % The weight matrix has dimension Kxd where K is the number of labels and d
        % is the number of dimensions (RGB pixels) of each image. 
        d = size(X_batch,1);
        K = size(Y_batch,1);

        % Each value of the matrix is initialize to have Gaussian random values 
        % with zero mean and standard deviation 1
        stdD = 0.01;
        W = stdD*randn(K, d);
        b = stdD*randn(K, 1);

        % 2.3 Parameters setting
        %
        % The parameters that we can set are chosen accordingly to the lecture
        % notes
        GDparams.n_batch = n_batch;
        GDparams.eta = eta;
        GDparams.n_epochs = n_epochs;
        lambda = lambda_in;

        % 2.4 Mini batch gradient descent algorithm
        %
        % The main algorithm is applied. We need to output the costs and loss to
        % plot them later.
        if display == 2
            disp( sprintf(' Training network with %f epochs, eta of %f and batch sixe of %f',n_epochs, eta, n_batch))
        end
        [Wstar, bstar, val_loss, train_loss] = MiniBatchGD(X_batch, Y_batch, X_val, Y_val, GDparams, W, b, lambda,display);

        % 2.5 Accuracy test
        %
        % Using the test data, we want to check the performance of our netwoek
        acc = ComputeAccuracy(X_test, y_test, Wstar, bstar);
        if display == 2
            disp( sprintf(' The accuracy of the network is %f', (acc*100)))
        end

        % 2.6 Plotting training results
        %
        % We want to visualize the results of the trining. First we are plotting
        % the training and validation loss for each epoch. 
        if display == 2
            figure; 
            plot(val_loss);
            hold on
            plot(train_loss);
            legend('Validation data', 'Training data')
            title(sprintf( 'Validation and training loss per epoch for epochs: %f, batch size: %f, eta: %f and lambda: %f ', n_epochs, n_batch, eta, lambda))
            hold off

            % Now we want to visualize the weights as images
            % Using this piece of code given in the lab notes
            for i=1:10
                im = reshape(Wstar(i, :), 32, 32, 3);
                s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
                s_im{i} = permute(s_im{i}, [2, 1, 3]);
            end

            % Plot it
            figure;
            for i = 1:10
                subplot(2,5,i)
                imagesc(s_im{i})
                axis square
                colormap jet
                if i == 3
                    title(sprintf( 'Weights representation for epochs: %f, batch size: %f, eta: %f and lambda: %f ', n_epochs, n_batch, eta, lambda))
                end
            end
            

        end
    end

end

%%%%%%%%%%%%%%%
%%Functions%%
%%%%%%%%%%%%%%%

function [X, Y, y] = LoadBatch(filename)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   - X contains the image pixel data, has size dxN, is of type double or
%     single and has entries between 0 and 1. N is the number of images
%     (10000) and d the dimensionality of each image (3072=32×32×3).
%   - Y is K×N (K= # of labels = 10) and contains the one-hot representation
%     of the label for each image.
%   - y is a vector of length N containing the label for each image. A note
%     of caution. CIFAR-10 encodes the labels as integers between 0-9 but
%     Matlab indexes matrices and vectors starting at 1. Therefore it may be
%     easier to encode the labels between 1-10.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    %Load the input batch
    In_batch = load(filename);

    %Number of labels
    K = 10;

    %Number of images 
    data_size = size(In_batch.data,1);

    %Matrix of images vectors (Normalized)
    X = double(In_batch.data')/255;

    %Lables changed from 0-9 to 1-10
    y = In_batch.labels + 1;

    %Inicializate the matrix of dimensions KxN
    Y = zeros(K,data_size);

    %Obtain the one-hot representation
    for i = 1:K
        rows = y == i;
        Y(i, rows) = 1;
    end

end


function P = EvaluateClassifier(X, W, b)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% where
%  X = each column of X corresponds to an image and it has size 
%       d(dimension of each image)xn(number of images).
%  W = matrix weights (number of labels)xd
%   b = bias (number of labels)
%   P = each column of P contains the probability for each label for the image
%       in the corresponding column of X. P has size Kxn.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For one image we have that
%   - s = Wx + b; 
%   - P = softmax(s)
    
    % First we prepare the dimension of b
    b = repmat(b,1,size(X,2));
    
    % Forward pass
    s = W*X + b;
    
    % Label probability for each image
    P = softmax(s);

end


function J = ComputeCost(X, Y, W, b, lambda)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% where
%  - X: each column of X corresponds to an image and X has size 
%        d(dimension of each image)xn(number of images).
%  - Y: each column of Y (K(number of labels)xn(number of images)) is the 
%        one-hot ground truth label for the corresponding column of X or Y 
%        is the (xn) vector of ground truth labels.
%  - J: is a scalar corresponding to the sum of the loss of the network's
%        predictions for the images in X relative to the ground truth labels and
%        the regularization term on W.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % We need the probabilities to compute the costs
    P = EvaluateClassifier( X, W, b );
    
    % Labelled training data
    D = size(X,2);
    
    % The equation for the cost is
    J = -1/D *sum(log(sum(Y.*P,1))) + lambda*sumsqr(W);

end

function acc = ComputeAccuracy(X, y, W, b)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% where
%  - X = each column of X corresponds to an image and X has size 
%         d(dimensions of each image)xn(number of images).
%  - y = is the vector of ground truth labels of length n.
%  - acc = is a scalar value containing the accuracy.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % We need the probabilities
    P = EvaluateClassifier( X, W, b);
    
    % We compute the vector of estimated labels for each picture.
    [~, indeces] = max(P);

    % Summatory of well classified images
    Total_correct = sum(indeces' == y);
    
    % Accuracy calculation
    acc = Total_correct/size(y,1);
     
end


function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% where
%  - X = each column of X corresponds to an image and it has size dxn.
%  - Y = each column of Y (Kxn) is the one-hot ground truth label for the 
%         corresponding column of X.
%  - P = each column of P contains the probability for each label for the image
%         in the corresponding column of X. P has size Kxn.
%  - grad_W =  is the gradient matrix of the cost J relative to W and has size
%               Kxd.
%  - grad_b = is the gradient vector of the cost J relative to b and has size
%              Kx1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This fucntion has been written following the indications included in the
% last slides of lecture 3. 

    % Initialize gradients
    grad_W = zeros(size(W));
    grad_b = zeros(size(W,1),1);

    % We need to iterate for each image, as we need to compute the gradiente
    % for each image. 
    for i=1:size(X, 2)
        % We calculate for every image the equation g.
        g = - Y(:,i)'/(Y(:,i)'*P(:,i)) * (diag(P(:,i)) - P(:, i)*P(:, i)');
        grad_b = grad_b + g';
        grad_W = grad_W + g'*X(:, i)';
    end

    % We have to divide the summatory between the batch size
    % Size of the batch
    B = size(X,2);

    grad_W = grad_W/B;
    grad_b = grad_b/B;

    % The last step is to add the regularization term
    grad_W = grad_W + 2*lambda*W;

end



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
            disp( sprintf(' Epoch: %f - Validation loss: %f - Training loss: %f', ep, val_loss(ep + 1), train_loss(ep + 1)))
        end

    end

    % We output the trained parameters
    Wstar = W;
    bstar = b;

end

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    no = size(W, 1);
    d = size(X, 1);

    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);

    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c1) / (2*h);
    end

    for i=1:numel(W)

        W_try = W;
        W_try(i) = W_try(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);

        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);

        grad_W(i) = (c2-c1) / (2*h);
    end

end