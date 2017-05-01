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