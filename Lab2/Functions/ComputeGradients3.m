%% ComputeGradient
function [ grad_W, grad_b ] = ComputeGradients3( X, Y, P, S, H, W, lambda )
%COMPUTEGRADIENTS Computes the gradients of the model parameters (W, b) for
%a mini-batch and using the softmax loss.
%
% Inputs:
%   X: Each column of X corresponds to an image, it has size (dxn)
%   Y: One-hot ground truth label for the corresponding image vector in X,
%       it has size (Kxn)
%   P: contains the probability for each label for the image 
%       in the corresponding column of X. It has size (Kxn)
%   W: Weight matrix, it has size (Kxd)
%   lambda: Weight on the regularization term
%   loss: Loss to be used, either softmax ("soft") or SVM ("svm")
%
% Outputs:
%   grad_W: Gradient of the Weight matrix, size (Kxd)
%   grad_b: Gradient of the bias vector, size (Kx1)

% Number of hidden layers
n_hidden = numel(W);
% Initialize outputs
grad_W = cell(1, n_hidden);
grad_b = cell(1, n_hidden);
% Merge X and hidden layers
X = [X H];

% Initialize gradients
for j=n_hidden:-1:1
    grad_W{j} = zeros(size(W{j}));
    grad_b{j} = zeros(1, size(W{j},1));
    %grad_b{j} = zeros(size(W{j},1), 1);
end

% Size of the batch
B = size(X{1}, 2);

for i=1:B
    y = Y(:,i);
    p = P(:, i);
    g = - y'/(y'*p) * (diag(p) - p*p');

    % For all layers (Propagate term g)
    for j=n_hidden:-1:1
        % Add gradient w.r.t. to model parameters for 
        grad_b{j} = grad_b{j} + g;
        grad_W{j} = grad_W{j} + g'*X{j}(:, i)';
        if j ~= 1
            g = g*W{j}*diag(S{j-1}(:, i)>0);
        end
    end
end



for j=n_hidden:-1:1
    % Normalize and add regularization term
    grad_W{j} = grad_W{j}/B + 2*lambda*W{j};
    grad_b{j} = grad_b{j}'/B;
end
end

