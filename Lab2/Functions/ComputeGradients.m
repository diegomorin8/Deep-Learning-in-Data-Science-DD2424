function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, hiddenEmissions, sHidden, lambda)

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
grad_W1 = zeros(size(W{1}));
grad_b1 = zeros(size(W{1},1),1);
grad_W2 = zeros(size(W{2}));
grad_b2 = zeros(size(W{2},1),1);

% We need to iterate for each image, as we need to compute the gradiente
% for each image. 
for i=1:size(X, 2)
    % We calculate for every image the equation g.
    g = - Y(:,i)'/(Y(:,i)'*P(:,i)) * (diag(P(:,i)) - P(:, i)*P(:, i)');
        
    % Gradient with respect to W2 and b2
    grad_b2 = grad_b2 + g';
    grad_W2 = grad_W2 + g'*hiddenEmissions(:, i)';
    
    % Propagate gradients
	g = g*W{2};
    
    s1 = sHidden(:,i)';
%     s1(s1 <= 0) = 0;
%     s1(s1 > 0) = 1;
    s1 = diag( s1 > 0 );
    
    % Gradient propagation
    g = g*s1;
    
    % Gradient with respect to W1 and b1
    grad_b1 = grad_b1 + g';
    grad_W1 = grad_W1 + g'*X(:, i)';
   
end

% We have to divide the summatory between the batch size
% Size of the batch
B = size(X,2);

grad_W1 = grad_W1/B;
grad_b{1} = grad_b1/B;
grad_W2 = grad_W2/B;
grad_b{2} = grad_b2/B;

% The last step is to add the regularization term
grad_W{1} = grad_W1 + 2*lambda*W{1};
% The last step is to add the regularization term
grad_W{2} = grad_W2 + 2*lambda*W{2};

end

