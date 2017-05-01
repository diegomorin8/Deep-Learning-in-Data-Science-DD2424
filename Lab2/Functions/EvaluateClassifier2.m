%% EvaluateClassifier
function [ P, S, H ] = EvaluateClassifier3( X, W, b )
% EVALUATECLASSIFIER   evaluates the scores of a batch of images
%   P = EvaluateClassifier( X, W, b ) performs the forward pass computing
%   the scores of each class for all data samples in X using the model
%   parameters W and b.
%
% Inputs:
%   X: Each column of X corresponds to an image, it has size (dxn)
%   W: Weight matrix, it has size (Kxd)
%   b: bias vector, it has size (Kx1)
%
% Outputs:
%   P: contains the probability for each label for the image 
%       in the corresponding column of X. It has size (Kxn)
%   S: contains the scpres for each label (class) for each image 
%       in the columns of X. It has size (Kxn)
%

n_hidden = numel(W);

S = cell(1, n_hidden);
H = cell(1, n_hidden-1);

% Iteratively multiply the data matrix X by the matrix of Weights W and add 
% the bias b (broadcasting)
S{1} = bsxfun(@plus,W{1}*X,b{1});

for i = 2:n_hidden
    H{i-1} = max(0, S{i-1});
    S{i} = bsxfun(@plus,W{i}*H{i-1},b{i});
end

P = bsxfun(@rdivide, exp(S{n_hidden}), sum(exp(S{n_hidden}), 1));

end
