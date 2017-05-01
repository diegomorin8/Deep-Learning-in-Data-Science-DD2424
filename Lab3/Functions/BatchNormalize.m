function s_norm = BatchNormalize( s, mu_, v )
    % Define a small eps
    eps = 1e-1000; 
    % Compute the normalization
    s_norm = ((diag(bsxfun(@plus, v, eps))^(0.5))*bsxfun(@minus, s, mu_));