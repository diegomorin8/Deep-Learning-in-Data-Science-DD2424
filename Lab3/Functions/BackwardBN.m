function [grad_W, grad_b] = BackwardBN(X, Y, P, W, h, s, s_norm, mu_, v, lambda)

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
    
    G = cell(1,size(W,2));
    % We need to iterate for each image, as we need to compute the gradient
    % for each image. 
    tic
    for i = 1:size(X, 2)
        
        % We calculate for every image the equation g.
        g = - Y(:,i)'/(Y(:,i)'*P(:,i)) * (diag(P(:,i)) - P(:, i)*P(:, i)');
        
        grad_b{size(W,2)} = grad_b{size(W,2)} + g';
        grad_W{size(W,2)} = grad_W{size(W,2)} + g'*h{size(W,2)}(:, i)';
        % Propagate gradients
        g = g*W{size(W,2)};

        s1 = s_norm{size(W,2)-1}(:,i)';
        s1 = diag( s1 > 0 );

        % Gradient propagation
        g = g*s1;
        all_g(:,i) = g;
    end
    G = all_g;
    % Gradient with respect to W2 and b2
    for j = (size(W,2)-1):-1:1
        G = BatchNormBackProp( G, s{j}, mu_{j}, v{j} );
        % Gradient with respect to W2 and b2
        
        if j > 1
            for i = 1:size(X, 2)
                
                g = G(:,i);
                grad_b{size(W,2)} = grad_b{size(W,2)} + g';
                grad_W{size(W,2)} = grad_W{size(W,2)} + g'*h{size(W,2)}(:, i)';
                % Propagate gradients
                g = g*W{j};
                s1 = s_norm{j-1}(:,i)';
                s1 = diag( s1 > 0 );
                % Gradient propagation
                g = g*s1;
                all_g(:,i) = g;
            end
            G = all_g; 
        end
    end
    
    % We have to divide the summatory between the batch size
    % Size of the batch
    B = size(X,2);

    for i = 1:size(W,2)
        % The last step is to add the regularization term
        grad_W{i} = grad_W{i}/B + 2*lambda*W{i};
        grad_b{i} = grad_b{i}/B;
    end

