function g_out = BatchNormBackProp( g, s, mu_, v)
    
    g_out = []; 
    dJ_dv = 0;
    dJ_dmu = 0; 
    % Define a small eps
    eps = 10e-3; 
    % Compute the normalization
    Vb = diag(bsxfun(@plus, v, eps));
    n = size(mu_, 1); 
    
    % First we define de
    g = g';
    dJ_dv = -(0.5)*g*(Vb^-1.5)*diag(bsxfun(@minus, s, mu_));
    dJ_dv = sum(dJ_dv,1);
    dJ_dmu = - g*(Vb^-0.5);
    dJ_dmu = sum(dJ_dmu,1);
    g_out = g*(Vb^-0.5) + (2/n)*(dJ_dv*diag(bsxfun(@minus, s, mu_))') + (1/n)*dJ_dmu;
    
% V2
% eps = 10e-10;
% h1 = (s - mu_)';
% sqrt_h2 = sqrt(v + eps);
% h2 = (1/sqrt(v + eps));
% corre = ((s-mu_)').^2;
% g = g';
% N = size(g,1);
% dh1 = h2*g;
% dinvvar = sum(h1*g,1);
% dsqrtvar = -(1/(sqrt_h2.^2))'*dinvvar;
% dvar = 0.5*(v + eps).^(-0.5)'*dsqrtvar;
% dcorre = (1/N)*ones(size(corre))'*dvar';
% dh1 = dh1 + 2 * h1 * dcorre;
% dmu = -sum(dh1,1);
% size(ones(size(dh1)))
% g_out = dh1 + (1/N)*ones(size(dh1))*dmu';
% size(h2)
% size(g)
% 
% 
% 
% g_out = 0;