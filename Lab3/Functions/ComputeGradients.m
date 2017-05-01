function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, h, s, lambda)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% where
%  - X = each column of X corresponds to an image and it has size dxn.
%  - Y = each column of Y (Kxn) is the one-hot ground truth label for the 
%         corresponding column of X.
%  - P = each column of P contains the probability for each label for the image
%         in the corresponding column of X. P has size Kxn.
%  - grad_W =  is the gradient matrix of the cost J relative to W and has size
%               Kxd.
%  - grad_b = is the gradient vector of the cost J relative to b and has size
%              Kx1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % This fucntion has been written following the indications included in the
    % last slides of lecture 3. 

    % Initialize gradients
    for i = 1:size(W,2)
        grad_W{i} = zeros(size(W{i}));
        grad_b{i} = zeros(size(W{i},1),1);

    end

    % We need to iterate for each image, as we need to compute the gradient
    % for each image. 
    for i=1:size(X, 2)

        % We calculate for every image the equation g.
        g = - Y(:,i)'/(Y(:,i)'*P(:,i)) * (diag(P(:,i)) - P(:, i)*P(:, i)');

        for j = size(W,2):-1:2
            % Gradient with respect to W2 and b2
            grad_b{j} = grad_b{j} + g';
            grad_W{j} = grad_W{j} + g'*h{j}(:, i)';
            
            % Propagate gradients
            g = g*W{j};

            s1 = s{j-1}(:,i)';
            s1 = diag( s1 > 0 );

            % Gradient propagation
            g = g*s1;
        end
        % Gradient with respect to W1 and b1
        grad_b{1} = grad_b{1} + g';
        grad_W{1} = grad_W{1} + g'*X(:, i)';

    end

    % We have to divide the summatory between the batch size
    % Size of the batch
    B = size(X,2);

    for i = 1:size(W,2)
        % The last step is to add the regularization term
        grad_W{i} = grad_W{i}/B + 2*lambda*W{i};
        grad_b{i} = grad_b{i}/B;
    end

