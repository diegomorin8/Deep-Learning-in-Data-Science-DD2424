%% ComputeAccuracy
function [ acc ] = ComputeAccuracy2( X, y, W, b )
% COMPUTEACCURACY  Computes the accuracy of the model on some dataset
%   acc = COMPUTEACCURACY(X, y, W, b) computes accuracy of the model
%   described by W and b on the set X with labels y.
%
% Inputs:
%   X: Each column of X corresponds to an image, it has size (dxn)
%   y: Ground truth labels for the corresponding image vectors in X,
%       it has size (nx1)
%   W: Weight matrix, it has size (Kxd)
%   b: bias vector, it has size (Kx1)
%
% Outputs:
%   acc: Accuracy obtained with the current model

% Obtain number of samples
n = size(X,2);

% Obtain scores
[P, ~, ~] = EvaluateClassifier( X, W, b );

% Obtain classification for each point
[~, idx] = max(P);

% Obtain accuracy
acc = sum(y == idx')/n;

end
