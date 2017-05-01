 %% initParans
function [W, b] = initParams(neurons_layer, n)
%INITPARAMS initializes the model parameters
%   [W, b] = initParams(neurons_layer, n)
%
% Inputs:
%   neurons_layer: vector containing neurons in each of the layers of the 
%                   network
%   n_samples: Number of samples
%
% Outputs:
%   W: Cell array containing the weight matrix of each layer
%   b: Cell array containing the bias vector of each layer


% Number of hidden layers
    n_hidden = numel(neurons_layer)-1;
    % Define cells for weight matrices and bias vectors
    W = cell(1,n_hidden);
    b = cell(1,n_hidden);
    % Standard deviation for weight matrices initialization, as proposed by
    % He et al. (201X)
    std_dev = sqrt(2.0/n);
    
    % Initialize each vector & matrix pair (for each layer)
    for i = 1:n_hidden
        rng(400);
        W{i} = std_dev*randn(neurons_layer(i+1), neurons_layer(i));
        b{i} = zeros(neurons_layer(i+1), 1); %std_dev*randn(neurons_layer(i+1), 1);
    end
end