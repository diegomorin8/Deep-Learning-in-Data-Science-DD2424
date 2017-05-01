function [W, b, mW, mb] = network_init2(DataDim, NumLabels, HIDDEN_NODES, SizeDataSet)

rng(400); 

if nargin < 3
    HIDDEN_NODES = 50;
end

W = cell(1,2);
b = cell(1,2);
mW = cell(1,2);
mb = cell(1,2);

% Each value of the matrix is initialize to have Gaussian random values 
% with zero mean and standard deviation given by Xavier initialization
stdD = sqrt(2.0/SizeDataSet);

rng(400);


% Initialize each vector & matrix pair (for each layer)
W{1} = stdD*randn(HIDDEN_NODES, DataDim);
W{2} = stdD*randn(NumLabels, HIDDEN_NODES);

b{1} = zeros(HIDDEN_NODES, 1);
b{2} = zeros(NumLabels,1);

mW{1} = zeros(size(W{1}));
mW{2} = zeros(size(W{2}));

mb{1} = zeros(size(b{1}));
mb{2} = zeros(size(b{2}));

