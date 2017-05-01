function [W, b, mW, mb] = network_init(layer_distribution, SizeDataSet)

    rng(400);

    if nargin < 3
        HIDDEN_NODES = 50;
    elseif nargin < 4
        SizeDataSet = 2; 
    elseif nargin < 5
        num_layers = 2; 
    end

    W = cell(1,size(layer_distribution,2)-1);
    b = cell(1,size(layer_distribution,2)-1);
    mW = cell(1,size(layer_distribution,2)-1);
    mb = cell(1,size(layer_distribution,2)-1);

    % Each value of the matrix is initialize to have Gaussian random values 
    % with zero mean and standard deviation given by Xavier initialization
    stdD = sqrt(2.0/SizeDataSet);

    rng(400);

    for i = 1:(size(layer_distribution,2)-1)
        W{i} = stdD*randn(layer_distribution(i+1),layer_distribution(i));
        b{i} = zeros(layer_distribution(i+1), 1);
        mW{i} = zeros(size(W{i}));
        mb{i} = zeros(size(b{i}));
    end
