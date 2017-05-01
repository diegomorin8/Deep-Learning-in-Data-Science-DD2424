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
     