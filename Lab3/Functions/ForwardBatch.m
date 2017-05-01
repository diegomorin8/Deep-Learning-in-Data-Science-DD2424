function [P, h, s, s_norm, mu_, v] = ForwardBatch(X, W, b)

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

    for i = 1:size(W,2)
        % First we prepare the dimension of b
        b{i} = repmat(b{i},1,size(X,2));
    end
    
    s = [];
    s_norm = [];
    h = []; 
    mu_ = []; 
    v = []; 
    
    h{1} = X; 
    for i = 1:(size(W,2) - 1)
        % Forward pass
        s{i} = W{i}*h{i} + b{i};
        
        n = size(s{i},2);
        %Compute mean and variance
        mu_{i} = mean(s{i},2);
        v{i} = var(s{i}, 0, 2)*(n-1) / n;
        % Normalize // CHECK QUE HAY QUE DEVOLVER
        s_norm{i} = BatchNormalize( s{i}, mu_{i}, v{i}); 
        % Hidden layer emission
        h{i+1} = max(0,s_norm{i});
    end
    
    % Forward pass
    s{size(W,2)} = W{size(W,2)}*h{size(W,2)} + b{size(W,2)};
    % Probability of each label
    P = softmax(s{size(W,2)});

end