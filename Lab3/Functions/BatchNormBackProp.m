function g_out = BatchNormBackProp( g, s, mu_, v)
    
    g_out = []; 
    % Define a small eps
    eps = 1e-10; 
    % Compute the normalization
    Vb = diag(bsxfun(@plus, v, eps));
    
    n = size(mu_, 1); 
    % First we define de gradients
    dJ_dv = -((0.5)*g*(Vb^1.5))'*diag(bsxfun(@minus, s, mu_))';
    dJ_dmu = -g*(Vb^0.5); 
    g_out = g*(Vb.^0.5) + (2/n)*(dJ_dv*diag(bsxfun(@minus, s, mu_)))' + (1/n)*dJ_dmu;