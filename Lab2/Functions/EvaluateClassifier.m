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