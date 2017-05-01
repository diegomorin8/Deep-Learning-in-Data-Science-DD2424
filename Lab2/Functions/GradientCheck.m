function GradientCheck(X_train, Y_train, W, b, lambda )
%GRADIENTCHECK Verifies that the analytical expression of the Gradient is
%   correct.
%
%   GradientCheck(X_train, Y_train, W, b, lambda )
%
% Inputs:
%   X_train: Each column of X corresponds to an image, it has size (dxn)
%   Y_train: One-hot ground truth label for the corresponding image vector 
%       in x_train, it has size (Kxn)
%   W: Weight matrix, it has size (Kxd)
%   b: bias vector, it has size (Kx1)

% We implement, analytically, the gradient computation. To verify that it
% is well implemented we run this "gradient check" test, which compares our
% results with the results obtained using numerical methods.

n_hidden = numel(W);

err_W = cell(n_hidden, 1);
err_b = cell(n_hidden, 1);

disp('Checking Gradient...');

% Runfor each batch
for i=1:100
    X_low = X_train(:,1+100*(i-1):100*i);
    Y_low = Y_train(:,1+100*(i-1):100*i);
    [P, S, H] = EvaluateClassifier2( X_low, W, b );
    [ grad_b, grad_W ] = ComputeGradients3( X_low, Y_low, P, S, H, W, ...
        lambda );
    [ ngrad_b, ngrad_W ] = ComputeGradsNumSlow3(X_low, Y_low, W, b, ...
        lambda, 1e-6);
    
    for j = 1:n_hidden
        err_W{j} = norm(grad_W{j}(:)-ngrad_W{j}(:))/...
        (norm(grad_W{j}(:))+norm(ngrad_W{j}(:)));
        err_b{j} = norm(grad_b{j}-ngrad_b{j})/(norm(grad_b{j})+norm(ngrad_b{j}));
        
        % Display warning if the difference is above a threshold
        if (err_W{j}>1e-6)
            fprintf('Weight gradient error of %d in layer %d! \n', err_W{j}, j);
        end
        if (err_b{j}>1e-6)
            fprintf('Bias gradient error of %d in layer %d! \n', err_b{j}, j);
        end

    end
    
    
end
