%%
% Load data sets
rng(400)

[X_batch, Y_batch, y_batch] = LoadBatch('data_batch_1.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

%% Pre - processing

mean_ = mean(X_batch, 2);
X_batch = X_batch - repmat(mean_, [1, size(X_batch, 2)]);
X_val = X_val - repmat(mean_, [1, size(X_val, 2)]);
X_test = X_test - repmat(mean_, [1, size(X_test, 2)]);

%%
% Number of features 
% Number of features 
[DataDim, ~] = size(X_batch);

% Number of nodes in the hidden layer
HIDDEN_NODES = [50 30]; 

% Number of labels
[NumLabels, ~] = size(Y_batch);

% Layers sizes
layer_distribution = [DataDim, HIDDEN_NODES,NumLabels];


% Size of the data set used in Xavier's initialization
SizeDataSet = size(X_batch,2);

[W, b, mW, mb] = network_init(layer_distribution, SizeDataSet);

%%
% The parameters are
GDparams.n_batch = 100;
GDparams.n_epochs = 10;
rho = 0.9;
eta_decay = 0.95;
etas = [];
lambdas = [];

% GDparams.eta = 0.0404;
% lambda = 0.0013;

% Funciona 1
% GDparams.eta = 0.02;
% lambda = 0.003;

% Funciona better
% GDparams.eta = 0.0404;
% lambda = 0.0013;

  
GDparams.eta = 0.0102;
lambda = 0;%eta_lambda_pairFinal(1,2);
[W_end, b_end, val_lossB1, train_lossB1] = MiniBatchGD(X_train, Y_train, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho, X_test, y_test);

%%
% The parameters are
GDparams.n_batch = 100;
GDparams.n_epochs = 10;
rho = 0.9;
eta_decay = 0.95;
etas = [];
lambdas = [];

% GDparams.eta = 0.0404;
% lambda = 0.0013;

% Funciona 1
% GDparams.eta = 0.02;
% lambda = 0.003;

% Funciona better
% GDparams.eta = 0.0404;
% lambda = 0.0013;

  
GDparams.eta = 0.0102;
eta3 = GDparams.eta; 
lambda = 0;%eta_lambda_pairFinal(1,2);
[W_end, b_end, val_lossNB1, train_lossNB1] = MiniBatchGD(X_train, Y_train, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho, X_test, y_test);

%%
% The parameters are
GDparams.n_batch = 100;
GDparams.n_epochs = 10;
rho = 0.9;
eta_decay = 0.95;
etas = [];
lambdas = [];

% GDparams.eta = 0.0404;
% lambda = 0.0013;

% Funciona 1
% GDparams.eta = 0.02;
% lambda = 0.003;

% Funciona better
% GDparams.eta = 0.0404;
% lambda = 0.0013;

  
GDparams.eta = 0.0102*10;
lambda = 0;%eta_lambda_pairFinal(1,2);
[W_end, b_end, val_lossNB2, train_lossNB2] = MiniBatchGD(X_train, Y_train, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho, X_test, y_test);

%%
% The parameters are
GDparams.n_batch = 100;
GDparams.n_epochs = 10;
rho = 0.9;
eta_decay = 0.95;
etas = [];
lambdas = [];

% GDparams.eta = 0.0404;
% lambda = 0.0013;

% Funciona 1
% GDparams.eta = 0.02;
% lambda = 0.003;

% Funciona better
% GDparams.eta = 0.0404;
% lambda = 0.0013;

  
GDparams.eta = 0.0102*10;
eta2 = GDparams.eta; 
lambda = 0;%eta_lambda_pairFinal(1,2);
[W_end, b_end, val_lossB2, train_lossB2] = MiniBatchGDNorm(X_train, Y_train, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho, X_test, y_test);


%%
% The parameters are
GDparams.n_batch = 100;
GDparams.n_epochs = 10;
rho = 0.9;
eta_decay = 0.95;
etas = [];
lambdas = [];

% GDparams.eta = 0.0404;
% lambda = 0.0013;

% Funciona 1
% GDparams.eta = 0.02;
% lambda = 0.003;

% Funciona better
% GDparams.eta = 0.0404;
% lambda = 0.0013;

  
GDparams.eta = 0.0102/10;
lambda = 0;%eta_lambda_pairFinal(1,2);
[W_end, b_end, val_lossNB3, train_lossNB3] = MiniBatchGD(X_train, Y_train, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho, X_test, y_test);

%%
% The parameters are
GDparams.n_batch = 100;
GDparams.n_epochs = 10;
rho = 0.9;
eta_decay = 0.95;
etas = [];
lambdas = [];

% GDparams.eta = 0.0404;
% lambda = 0.0013;

% Funciona 1
% GDparams.eta = 0.02;
% lambda = 0.003;

% Funciona better
% GDparams.eta = 0.0404;
% lambda = 0.0013;

  
GDparams.eta = 0.0102/10;
eta3 = GDparams.eta; 
lambda = 0;%eta_lambda_pairFinal(1,2);
[W_end, b_end, val_lossB3, train_lossB3] = MiniBatchGDNorm(X_train, Y_train, X_val, Y_val, GDparams, W, b, lambda, 2, eta_decay, mW, mb, rho, X_test, y_test);

%%
figure
subplot(1,3,1)
plot(val_lossB1)
hold on
plot(val_lossNB1)
hold on
plot(train_lossB1)
hold on
plot(train_lossNB1)
hold on
legend( ' Validation loss with BN ' ,' Validation loss without BN ' ,' Trining loss with BN ' ,' Training loss without BN ')
title(sprintf(' Eta = %g', eta1))
subplot(1,3,2)
plot(val_lossB2)
hold on
plot(val_lossNB2)
hold on
plot(train_lossB2)
hold on
plot(train_lossNB2)
hold on
legend( ' Validation loss with BN ' ,' Validation loss without BN ' ,' Trining loss with BN ' ,' Training loss without BN ')
title(sprintf(' Eta = %g', eta2))
subplot(1,3,3)
plot(val_lossB3)
hold on
plot(val_lossNB3)
hold on
plot(train_lossB3)
hold on
plot(train_lossNB3)
hold on
legend( ' Validation loss with BN ' ,' Validation loss without BN ' ,' Trining loss with BN ' ,' Training loss without BN ')
title(sprintf(' Eta = %g', eta3))