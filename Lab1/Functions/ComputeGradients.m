function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% where
%  - X = each column of X corresponds to an image and it has size dxn.
%  - Y = each column of Y (Kxn) is the one-hot ground truth label for the 
%         corresponding column of X.
%  - P = each column of P contains the probability for each label for the image
%         in the corresponding column of X. P has size Kxn.
%  - grad_W =  is the gradient matrix of the cost J relative to W and has size
%               Kxd.
%  - grad_b = is the gradient vector of the cost J relative to b and has size
%              Kx1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This fucntion has been written following the indications included in the
% last slides of lecture 3. 


 
% Initialize gradients
grad_W = zeros(size(W));
grad_b = zeros(size(W,1),1);

% We need to iterate for each image, as we need to compute the gradiente
% for each image. 
for i=1:size(X, 2)
    % We calculate for every image the equation g.
    g = - Y(:,i)'/(Y(:,i)'*P(:,i)) * (diag(P(:,i)) - P(:, i)*P(:, i)');
    grad_b = grad_b + g';
    grad_W = grad_W + g'*X(:, i)';
end

% We have to divide the summatory between the batch size
% Size of the batch
B = size(X,2);

grad_W = grad_W/B;
grad_b = grad_b/B;

% The last step is to add the regularization term
grad_W = grad_W + 2*lambda*W;

end

