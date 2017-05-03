function [W_end, b_end, val_loss, train_loss, accuracy ] = ass1_main( n_batch, eta, n_epochs, lambda_in, rho, eta_decay, all_data, display )

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
    
    % Set the seed
    rng(400);
    if all_data == 1
        % Load all data sets
        [ X_train, Y_train, y_train ] = LoadBatch( 'data_batch_1.mat' );
        [ X_val, Y_val, y_val ] = LoadBatch( 'data_batch_2.mat' );
        [ X_train2, Y_train2, y_train2 ] = LoadBatch( 'data_batch_3.mat' );
        [ X_train3, Y_train3, y_train3 ] = LoadBatch( 'data_batch_4.mat' );
        [ X_train4, Y_train4, y_train4 ] = LoadBatch( 'data_batch_5.mat' );
        [ X_test, Y_test, y_test ] = LoadBatch( 'test_batch.mat' );

        % We take 9000 of the samples from the validation set
        X_train = [X_train, X_val(:, 1:9000), X_train2, X_train3, X_train4];
        Y_train = [Y_train, Y_val(:, 1:9000), Y_train2, Y_train3, Y_train4];
        y_train = [y_train; y_val(1:9000); y_train2; y_train3; y_train4];
        clear X_train2 X_train3 X_train4
        clear Y_train2 Y_train3 Y_train4
        clear y_train2 y_train3 y_train4

        X_val = X_val(:, 9001:end);
        Y_val = Y_val(:, 9001:end);
        y_val = y_val(9001:end);

        mean_ = mean(X_train, 2);
        X_train = X_train - repmat(mean_, [1, size(X_train, 2)]);
        X_val = X_val - repmat(mean_, [1, size(X_val, 2)]);
        X_test = X_test - repmat(mean_, [1, size(X_test, 2)]);
       
        % Number of features 
        [DataDim, ~] = size(X_batch);

        % Number of nodes in the hidden layer
        HIDDEN_NODES = [50 30]; 

        % Number of labels
        [NumLabels, ~] = size(Y_batch);

        % Layers sizes
        layer_distribution = [DataDim, HIDDEN_NODES,NumLabels];


        % Size of the data set used in Xavier's initialization
        SizeDataSet = size(X_batch,2);

        [W, b, mW, mb] = network_init(layer_distribution, SizeDataSet);

        % The parameters are
        GDparams.n_batch = n_batch;
        GDparams.n_epochs = n_epochs;
        rho = rho;
        eta_decay = eta_decay;

        GDparams.eta = eta;
        lambda = lambda_in;

        % Train the network
        [W_end, b_end, val_loss, train_loss] = MiniBatchGDNorm(X_train, Y_train, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho, X_test, y_test);
        accuracy = ComputeAccuracy(X_test, y_test, W_end, b_end);
        % Plot
        figure;
        plot(val_loss)
        hold on
        plot(train_loss)
        title(sprintf( 'Validation vs training loss for eta = %g - lambda = %g. Final accuracy = %g', eta, lambda, ComputeAccuracy( X_test, y_test, W_end, b_end))) 
        legend('Validation loss', 'Training loss' )
        hold off

    % Only one batch
    elseif all_data == 0

        [X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
        [X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
        [X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

        % Pre - processing

        mean_ = mean(X_train, 2);
        X_train = X_train - repmat(mean_, [1, size(X_train, 2)]);
        X_val = X_val - repmat(mean_, [1, size(X_val, 2)]);
        X_test = X_test - repmat(mean_, [1, size(X_test, 2)]);

        % Number of features 
        [DataDim, ~] = size(X_batch);

        % Number of nodes in the hidden layer
        HIDDEN_NODES = [50 30]; 

        % Number of labels
        [NumLabels, ~] = size(Y_batch);

        % Layers sizes
        layer_distribution = [DataDim, HIDDEN_NODES,NumLabels];


        % Size of the data set used in Xavier's initialization
        SizeDataSet = size(X_batch,2);

        [W, b, mW, mb] = network_init(layer_distribution, SizeDataSet);

        % The parameters are
        GDparams.n_batch = n_batch;
        GDparams.n_epochs = n_epochs;
        rho = rho;
        eta_decay = eta_decay;

        GDparams.eta = eta;
        lambda = lambda_in;

        % Train the network
        [W_end, b_end, val_loss, train_loss] = MiniBatchGDNorm(X_train, Y_train, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho, X_test, y_test);
        accuracy = ComputeAccuracy(X_test, y_test, W_end, b_end);

        % Plot
        figure;
        plot(val_loss)
        hold on
        plot(train_loss)
        title(sprintf( 'Validation vs training loss for eta = %g - lambda = %g. Final accuracy = %g', eta, lambda, ComputeAccuracy( X_test, y_test, W_end, b_end))) 
        legend('Validation loss', 'Training loss' )
        hold off
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


function [P,h,s1] = EvaluateClassifier(X, W, b)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% where
% - X = each column of X corresponds to an image and it has size 
%       d(dimension of each image)xn(number of images).
% - W = matrix weights (number of labels)xd
%   b = bias (number of labels)
%   P = each column of P contains the probability for each label for the image
%       in the corresponding column of X. P has size Kxn.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For one image we have that
% s1 = W1x + b1 (1)
% h = max(0, s1) (2)
% s = W2h + b2 (3)
% p = SOFTMAX(s) (4)

    
    % First we prepare the dimension of b
    b1 = repmat(b{1},1,size(X,2));
    b2 = repmat(b{2},1,size(X,2));
    
    % Forward pass
    s1 = W{1}*X + b1;
    
    % Hidden layer emission
    h = max(0,s1);
    
    % Forward pass
    s2 = W{2}*h + b2;
    
    % Probability of each label
    P = softmax(s2);
    

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
    [P,~,~] = EvaluateClassifier( X, W, b);
    
    % Labelled training data
    D = size(X,2);
    
    % The equation for the cost is
    J = -1/D *sum(log(sum(Y.*P,1))) + lambda*(sumsqr(W{1}) + sumsqr(W{2}));
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
    [P,~,~] = EvaluateClassifier( X, W, b);
    
    % We compute the vector of estimated labels for each picture.
    [~, indeces] = max(P);

    % Summatory of well classified images
    Total_correct = sum(indeces' == y);
    
    % Accuracy calculation
    acc = Total_correct/size(y,1);

end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, hiddenEmissions, sHidden, lambda)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% where
%  - X = each column of X corresponds to an image and it has size dxn.
%  - Y = each column of Y (Kxn) is the one-hot ground truth label for the 
%         corresponding column of X.
%  - P = each column of P contains the probability for each label for the image
%         in the corresponding column of X. P has size Kxn.
%  - grad_W =  is the gradient matrix of the cost J relative to W and has size
%               Kxd.
%  - grad_b = is the gradient vector of the cost J relative to b and has size
%              Kx1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % This function has been written following the indications included in the
    % last slides of lecture 3. 

    % Initialize gradients
    grad_W1 = zeros(size(W{1}));
    grad_b1 = zeros(size(W{1},1),1);
    grad_W2 = zeros(size(W{2}));
    grad_b2 = zeros(size(W{2},1),1);

    % We need to iterate for each image, as we need to compute the gradiente
    % for each image. 
    for i=1:size(X, 2)
        % We calculate for every image the equation g.
        g = - Y(:,i)'/(Y(:,i)'*P(:,i)) * (diag(P(:,i)) - P(:, i)*P(:, i)');

        % Gradient with respect to W2 and b2
        grad_b2 = grad_b2 + g';
        grad_W2 = grad_W2 + g'*hiddenEmissions(:, i)';

        % Propagate gradients
        g = g*W{2};

        s1 = sHidden(:,i)';
        s1 = diag( s1 > 0 );

        % Gradient propagation
        g = g*s1;

        % Gradient with respect to W1 and b1
        grad_b1 = grad_b1 + g';
        grad_W1 = grad_W1 + g'*X(:, i)';

    end

    % We have to divide the summatory between the batch size
    % Size of the batch
    B = size(X,2);

    grad_W1 = grad_W1/B;
    grad_b{1} = grad_b1/B;
    grad_W2 = grad_W2/B;
    grad_b{2} = grad_b2/B;

    % The last step is to add the regularization term
    grad_W{1} = grad_W1 + 2*lambda*W{1};
    % The last step is to add the regularization term
    grad_W{2} = grad_W2 + 2*lambda*W{2};

end


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

                % Update network parameters as in the lecture notes
                W{1} = W{1} - eta*Grad_W{1};
                W{2} = W{2} - eta*Grad_W{2};
                b{1} = b{1} - eta*Grad_b{1};
                b{2} = b{2} - eta*Grad_b{2};

            elseif moments_check == 1

                 % Update network parameters as in the lecture notes
                mW{1} = rho*mW{1} + eta*Grad_W{1};
                mW{2} = rho*mW{2} + eta*Grad_W{2};
                mb{1} = rho*mb{1} + eta*Grad_b{1};
                mb{2} = rho*mb{2} + eta*Grad_b{2};

                W{1} = W{1} - mW{1};
                W{2} = W{2} - mW{2};
                b{1} = b{1} - mb{1};
                b{2} = b{2} - mb{2};

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

function checkGrad_numeric()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function calculate the numeric and analytic gradients and computes
% the difference between then. Then plots the error. The error must be
% below 1E-6 to considered that the function that computes the gradients
% analytically works well.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
            
            % Show the iteration number
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
        % Plot the error
        figure;
        plot(error_W)
        hold on
        plot(error_b)
        legend( 'Error in W', ' Error in B')
        axis([ 0 100 0 1e-8 ])
end

function checkGrad_overfit()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function does a snity check on the function that computes gradients 
% It trains the network during 200 epochs using a small subset of one of the 
% dataset's batches. If the gradients are calculated correctly, the network 
% must be able to overfit the training data.  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

    rng(400)
    [X_batch, Y_batch, y_batch] = LoadBatch('data_batch_1.mat');
    [X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
    [X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

    % Pre - processing

    mean_ = mean(X_batch, 2);
    X_batch = X_batch - repmat(mean_, [1, size(X_batch, 2)]);
    X_val = X_val - repmat(mean_, [1, size(X_val, 2)]);
    X_test = X_test - repmat(mean_, [1, size(X_test, 2)]);

    % Weights initialization

    lambda = 0; 

    % Number of features 
    [DataDim, ~] = size(X_batch);

    % Number of nodes in the hidden layer
    HIDDEN_NODES = 50;

    % Number of labels
    [NumLabels, ~] = size(Y_batch);

    % Size of the data set used in Xavier's initialization
    SizeDataSet = size(X_batch,2);

    % Weight and bias initialization
    [W, b, mW, mb] = network_init(DataDim, NumLabels, HIDDEN_NODES, SizeDataSet);

    % The parameters that we can set are chosen accordingly to the lecture
    % notes
    GDparams.n_batch = 100;
    GDparams.eta = 0.35;
    GDparams.n_epochs = 200;
    lambda = 0;

    %
    % The main algorithm is applied. We need to output the costs and loss to
    % plot them later. We are using just a small subset of the dataset
    tic
    [Wstar, bstar, val_loss_noM, train_loss_noM] = MiniBatchGD(X_batch(:,1:100), Y_batch(:,1:100), X_val(:,1:100), Y_val(:,1:100), GDparams, W, b, lambda, 2);
    time_noM = toc;

    % Check performance of gradient descent - Accuracy test
    %
    % Using the test data, we want to check the performance of our netwoek
    acc_noM = ComputeAccuracy(X_test, y_test, Wstar, bstar)

    % That of course, is really low.

    % 2.7.2 Rho = 0.5 - No decay
    % 
    % The first step is to prepare the workspace

    %
    % The parameters that we can set are chosen accordingly to the lecture
    % notes
    GDparams.n_batch = 100;
    GDparams.eta = 0.35;
    GDparams.n_epochs = 200;
    rho = 0.5;
    eta_decay = 1;
    lambda = 0;

    %
    %
    % The main algorithm is applied. We need to output the costs and loss to
    % plot them later. We are using just a small subset of the dataset
    tic
    [Wstar,bstar, val_loss_rho5, train_loss_rho5] = MiniBatchGD(X_batch(:,1:100), Y_batch(:,1:100), X_val(:,1:100), Y_val(:,1:100), GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho);
    time_rho5 = toc;
    % Check performance of gradient descent - Accuracy test
    %
    % Using the test data, we want to check the performance of our netwoek
    acc_rho5 = ComputeAccuracy(X_test, y_test, Wstar, bstar)

    % That of course, is really low. 

    % 2.7.3 Rho = 0.9 - No decay
    % 
    % 
    %
    % The parameters that we can set are chosen accordingly to the lecture
    % notes
    GDparams.n_batch = 100;
    GDparams.eta = 0.35;
    GDparams.n_epochs = 200;
    rho = 0.95;
    eta_decay = 1;
    lambda = 0;

    %
    %
    % The main algorithm is applied. We need to output the costs and loss to
    % plot them later. We are using just a small subset of the dataset
    tic
    [Wstar,bstar, val_loss_rho95, train_loss_rho95] = MiniBatchGD(X_batch(:,1:100), Y_batch(:,1:100), X_val(:,1:100), Y_val(:,1:100), GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho);
    time_rho95 = toc;
    % Check performance of gradient descent - Accuracy test
    %
    % Using the test data, we want to check the performance of our netwoek
    acc_rho95 = ComputeAccuracy(X_test, y_test, Wstar, bstar)

    % That of course, is really low. 


    % 2.7.4 Rho = 0.99 - No decay
    % 

    %
    % The parameters that we can set are chosen accordingly to the lecture
    % notes
    GDparams.n_batch = 100;
    GDparams.eta = 0.35;
    GDparams.n_epochs = 200;
    rho = 0.99;
    eta_decay = 1;
    lambda = 0;

    %
    %
    % The main algorithm is applied. We need to output the costs and loss to
    % plot them later. We are using just a small subset of the dataset
    tic
    [Wstar, bstar, val_loss_rho99, train_loss_rho99] = MiniBatchGD(X_batch(:,1:100), Y_batch(:,1:100), X_val(:,1:100), Y_val(:,1:100), GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho);
    time_rho99 = toc;

    % Check performance of gradient descent - Accuracy test
    %
    % Using the test data, we want to check the performance of our netwoek
    acc_rho99 = ComputeAccuracy(X_test, y_test, Wstar, bstar)

    % That of course, is really low. 

    % 2.7.5 Rho = 0.5 - Decay 0.95
    % 
    % % We want to clean the work space to start the experiments again
    %
    %
    % The parameters that we can set are chosen accordingly to the lecture
    % notes
    GDparams.n_batch = 100;
    GDparams.eta = 0.35;
    GDparams.n_epochs = 200;
    rho = 0.5;
    eta_decay = 0.95;
    lambda = 0;

    %
    %
    % The main algorithm is applied. We need to output the costs and loss to
    % plot them later. We are using just a small subset of the dataset
    tic
    [Wstar, bstar val_loss_rho5Dec train_loss_rho5Dec] = MiniBatchGD(X_batch(:,1:100), Y_batch(:,1:100), X_val(:,1:100), Y_val(:,1:100), GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho);
    time_rho5Dec = toc;
    % Check performance of gradient descent - Accuracy test
    %
    % Using the test data, we want to check the performance of our netwoek
    acc_rho5Dec = ComputeAccuracy(X_test, y_test, Wstar, bstar)

    % That of course, is really low. 

    % 2.7.6 Rho = 0.9 - Decay = 0.95
    % 
    % % We want to clean the work space to start the experiments again

    %
    % The parameters that we can set are chosen accordingly to the lecture
    % notes
    GDparams.n_batch = 100;
    GDparams.eta = 0.35;
    GDparams.n_epochs = 200;
    rho = 0.95;
    eta_decay = 0.95;
    lambda = 0;

    %
    %
    % The main algorithm is applied. We need to output the costs and loss to
    % plot them later. We are using just a small subset of the dataset
    tic
    [Wstar,bstar, val_loss_rho95Dec, train_loss_rho95Dec] = MiniBatchGD(X_batch(:,1:100), Y_batch(:,1:100), X_val(:,1:100), Y_val(:,1:100), GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho);
    time_rho95Dec = toc;

    % Check performance of gradient descent - Accuracy test
    %
    % Using the test data, we want to check the performance of our netwoek
    acc_rho95Dec = ComputeAccuracy(X_test, y_test, Wstar, bstar)

    % That of course, is really low. 

    % 2.7.7 Results Comparison

    % We want to visualize the results of the trining. First we are plotting
    % the training and validation loss for each epoch. 

    figure; 
    subplot(3,2,1)
    plot(val_loss_noM);
    hold on
    plot(train_loss_noM);
    axis([0 200 0 4])
    legend('Validation data', 'Training data')
    title(sprintf(' Validation - Trining loss - eta = %g - rho = %g - decay = %g - time = %g', GDparams.eta, 0, 1, time_noM))
    hold off
    subplot(3,2,2)
    plot(val_loss_rho5);
    hold on
    plot(train_loss_rho5);
    axis([0 200 0 4])
    legend('Validation data', 'Training data')
    title(sprintf(' Validation - Trining loss - eta = %g - rho = %g - decay = %g - time = %g', GDparams.eta, 0.5, 1, time_rho5))
    hold off
    subplot(3,2,3)
    plot(val_loss_rho95);
    hold on
    plot(train_loss_rho95);
    axis([0 200 0 4])
    legend('Validation data', 'Training data')
    title(sprintf(' Validation - Trining loss - eta = %g - rho = %g - decay = %g - time = %g', GDparams.eta, 0.95, 1, time_rho95))
    hold off
    subplot(3,2,4)
    plot(val_loss_rho99);
    hold on
    plot(train_loss_rho99);
    axis([0 200 0 4])
    legend('Validation data', 'Training data')
    title(sprintf(' Validation - Trining loss - eta = %g - rho = %g - decay = %g - time = %g', GDparams.eta, 0.99, 1, time_rho95))
    hold off
    subplot(3,2,5)
    plot(val_loss_rho5Dec);
    hold on
    plot(train_loss_rho5Dec);
    axis([0 200 0 4])
    legend('Validation data', 'Training data')
    title(sprintf(' Validation - Trining loss - eta = %g - rho = %g - decay = %g - time = %g', GDparams.eta, 0.5, 0.95, time_rho5Dec))
    hold off
    subplot(3,2,6)
    plot(val_loss_rho95Dec);
    hold on
    plot(train_loss_rho95Dec);
    axis([0 200 0 4])
    legend('Validation data', 'Training data')
    title(sprintf(' Validation - Trining loss - eta = %g - rho = %g - decay = %g - time = %g', GDparams.eta, 0.95, 0.95, time_rho95Dec))
    hold off

end

function eta_range()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimates and plot the optimal coarse range for eta 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % 3.1 Reasonable range of values for the learning rate
    %
    % Steps: 
    %   1. Set the regularization term to a small value (say .000001.)

    %   2. Randomly initalized the network. It should perform similarly to a random
    %      guesser and thus the training loss before you start learning should 
    %      be ?2.3 (? ln(.1)). 

    % Very low lambda
    lambda = 0.000001;
    rng(400)
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

    % Weights initialization

    % Number of features 
    [DataDim, ~] = size(X_batch);

    % Number of nodes in the hidden layer
    HIDDEN_NODES = 50;

    % Number of labels
    [NumLabels, ~] = size(Y_batch);

    % Size of the data set used in Xavier's initialization
    SizeDataSet = size(X_batch,2);

    % Weight and bias initialization
    [W, b, mW, mb] = network_init(DataDim, NumLabels, HIDDEN_NODES, SizeDataSet);
    %
    %   3. You want to perform a quick search by hand to find the rough bounds 
    %      for reasonable values of the learning rate. During this stage you 
    %      should print out the training cost/lost after every epoch. If the 
    %      learning rate is too small, then after each epoch you will find 
    %      that the training loss barely changes. While if the learning rate is 
    %      too large, then learning is unstable and you will probably get 
    %      NaNs and/or very high loss values. You should only need ?5 epochs to
    %      check these properties.


    % The parameters are
    GDparams.n_batch = 100;
    GDparams.n_epochs = 2;
    rho = 0.9;
    eta_decay = 0.95;
    valLoss = [];
    trainLoss = [];
    index = [];
    cont = 0;
    str = [];
    
    % Try all these values of eta
    for i = 0.01:0.02:0.16
        cont = cont + 1;
        disp( cont )
        
        % Store eta in GD params
        GDparams.eta = i;
        
        % Train the network (dont save the weights nor bias)
        [~, ~, val_loss, train_loss] = MiniBatchGD(X_batch, Y_batch, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho);
        
        %Store values
        valLoss = [valLoss , val_loss];
        trainLoss = [trainLoss , train_loss];
        index = [index ; i];
        str = [str ; sprintf('eta = %g',i)];
    end

    % Plot
    figure; 
    plot(trainLoss);
    legend(str)
    hold off
end




function eta_lambda_random_search()
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This calculates the optimal parameter tandem of lambda and eta and
% displays them
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
    rng(400);
    
    % Number of features 
    [DataDim, ~] = size(X_batch);

    % Number of nodes in the hidden layer
    HIDDEN_NODES = 50;

    % Number of labels
    [NumLabels, ~] = size(Y_batch);

    % Size of the data set used in Xavier's initialization
    SizeDataSet = size(X_batch,2);

    % Weight and bias initialization
    [W, b, mW, mb] = network_init(DataDim, NumLabels, HIDDEN_NODES, SizeDataSet);

    %Valid ranges
    minEta = 0.005;
    maxEta = 0.3;

    minL = 0.0005;
    maxL = 0.01;
    
    % First random search
    
    % The parameters are
    GDparams.n_batch = 100;
    GDparams.n_epochs = 1;
    rho = 0.9;
    eta_decay = 0.95;
    valLoss1 = [];
    trainLoss1 = [];
    etas1 = [];
    lambdas1 = [];
    accuracy1 = [];
    cont  = 0;
    for i = 1:60
        % Number of iterations
        cont = cont + 1;
        
        % Random picked eta
        e = minEta + (maxEta - minEta)*rand(1, 1); 
        
        % Save the eta
        etas1 = [etas1 ; e];
        GDparams.eta = etas1(i);
        
        % Random picked lambda
        l = minL + (maxL - minL)*rand(1, 1);
        
        % Save tha lambda
        lambdas1 = [lambdas1 ; l];
        lambda = lambdas1(i);
        
        % Train the network
        [W_end, b_end, val_loss, train_loss] = MiniBatchGD(X_batch, Y_batch, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho, X_test, y_test);
        
        % Save importan values
        valLoss1 = [valLoss1 , val_loss];
        trainLoss1 = [trainLoss1 , train_loss];
        acc = ComputeAccuracy(X_val, y_val, W_end, b_end);
        accuracy1 = [accuracy1, acc] ; 
        disp(sprintf('Cont = %i -  Accuracy = %g - Eta = %g - Lambda = %g', cont, accuracy1(i), etas1(i), lambdas1(i)))
    end

    % Compute the first coarse range for lambda and eta
    % The winners will be the one with higher accuracy on the validation
    % set
    [ accVal1, accInd1] = sort( accuracy1, 'descend' ); 

    % Pick the 15 best
    N = 15; 
    
    % Save the indeces
    accInd1 = accInd1(1:N);
    
    % Save the winner pairs
    eta_lambda_pair1 = [ etas1(accInd1) , lambdas1(accInd1)];

    % Set the limits for the next random search
    maxL1 = max(lambdas1(accInd1));
    minL1 = min(lambdas1(accInd1));
    maxEta1 = max(etas1(accInd1));
    minEta1 = min(etas1(accInd1));
    
    % Show the result
    disp( sprintf ( ' The boundaries are eta = %g - %g, lambda = %g - %g ', minEta1, maxEta1, minL1, maxL1))
    % Restart the seed
    rng(400) 

    % The parameters are
    GDparams.n_batch = 100;
    GDparams.n_epochs = 3;
    rho = 0.9;
    eta_decay = 0.95;
    valLoss2 = [];
    trainLoss2 = [];
    etas2 = [];
    lambdas2 = [];
    accuracy2 = [];
    cont = 0; 
    % Bigger range to make sure that the extreme values are included
    minEta1 = minEta1 * 0.1;
    minL1 = minL1 * 0.1;
    
    for i = 1:60
        % Number of iterations
        cont = cont + 1 ;
        
        % Random pick of eta
        e = minEta1 + (maxEta1 - minEta1)*rand(1, 1);
        
        % Save eta
        etas2 = [etas2 ; e];
        GDparams.eta = etas2(i);
        
        % Random pick of lambda
        l = minL1 + (maxL1 - minL1)*rand(1, 1);
        
        % Save lambda
        lambdas2 = [lambdas2 ; l];
        lambda = lambdas2(i);
        
        % Train the network
        [W_end, b_end,  val_loss, train_loss] = MiniBatchGD(X_batch, Y_batch, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho, X_test, y_test);
        
        % Store important data
        valLoss2 = [valLoss2 , val_loss];
        trainLoss2 = [trainLoss2 , train_loss];
        accuracy2 = [accuracy2, ComputeAccuracy(X_val, y_val, W_end, b_end)] ; 
        disp(sprintf('Cont = %i -  Accuracy = %g - Eta = %g - Lambda = %g', cont, accuracy2(i), etas2(i), lambdas2(i)))
    end

    % The winner pairs will be the one with higher accuracies on the
    % validation data
    [ accVal2, accInd2] = sort( accuracy2, 'descend' ); 

    % Pick only the 8 best
    N = 8; 
    
    % Save the indexes
    accInd2 = accInd2(1:N);

    % Save the winner pairs
    eta_lambda_pair2 = [ etas2(accInd2) , lambdas2(accInd2)];

    % Boundaries
    maxL2 = max(lambdas2(accInd2));
    minL2 = min(lambdas2(accInd2));
    maxEta2 = max(etas2(accInd2));
    minEta2 = min(etas2(accInd2));
    
    disp( sprintf ( ' The boundaries are eta = %g - %g, lambda = %g - %g ', minEta2, maxEta2, minL2, maxL2))
    disp(' The best pairs are ')
    disp(eta_lambda_pair2)
end


function [grad_W, grad_b] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);

    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));

        for i=1:length(b{j})

            b_try = b;
            b_try{j}(i) = b_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W, b_try, lambda);

            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W, b_try, lambda);

            grad_b{j}(i) = (c2-c1) / (2*h);
        end
    end

    for j=1:length(W)
        grad_W{j} = zeros(size(W{j}));

        for i=1:numel(W{j})

            W_try = W;
            W_try{j}(i) = W_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W_try, b, lambda);

            W_try = W;
            W_try{j}(i) = W_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W_try, b, lambda);

            grad_W{j}(i) = (c2-c1) / (2*h);
        end
    end
end