%% ComputeCost
function [ J ] = ComputeCost2( X, Y, W, b, lambda )
%COMPUTECOST Computes the cost function for a set of images
%   J = ComputeCost( X, Y, W, b, lambda ) computes the cost on the set X 
%   with labels y of the model described by parameters W and b, where 
%   lambda is the regularization strength.
%
% Inputs:
%   X: Each column of X corresponds to an image, it has size (dxn)
%   Y: One-hot ground truth label for the corresponding image vector in X,
%       it has size (Kxn)
%   W: Weight matrix, it has size (Kxd)
%   b: bias vector, it has size (Kx1)
%   lambda: Weight on the regularization term
%   loss: Loss to be used, either softmax ("soft") or SVM ("svm")
%
% Outputs:
%   J: Cost obtained after adding the loss of the network predictions for 
%       images in X. It is a (scalar)

% Obtain class probabilities of each sample in X
[P, ~, ~] = EvaluateClassifier2( X, W, b );

% Obtain the regularization term
reg = 0;

for i = 1:numel(W)
    reg = reg + sumsqr(W{i});
end

% Combine regularization term with cros-validation loss
D = size(X,2);

J = -1/D *sum(log(sum(Y.*P,1))) + lambda*reg;


end
