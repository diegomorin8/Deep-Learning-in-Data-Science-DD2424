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
    [P,~,~] = ForwardBatch( X, W, b);
    
    % Labelled training data
    D = size(X,2);
    
    sumsqrTotal = 0; 
    for i = 1:size(W,2)
        sumsqrTotal = sumsqrTotal + sumsqr(W{i});
    end
    
    % The equation for the cost is
    J = -1/D *sum(log(sum(Y.*P,1))) + lambda*sumsqrTotal;