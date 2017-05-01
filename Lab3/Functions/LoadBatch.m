function [X, Y, y] = LoadBatch(filename)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   - X contains the image pixel data, has size dxN, is of type double or
%     single and has entries between 0 and 1. N is the number of images
%     (10000) and d the dimensionality of each image (3072=32×32×3).
%   - Y is K×N (K= # of labels = 10) and contains the one-hot representation
%     of the label for each image.
%   - y is a vector of length N containing the label for each image. A note
%     of caution. CIFAR-10 encodes the labels as integers between 0-9 but
%     Matlab indexes matrices and vectors starting at 1. Therefore it may be
%     easier to encode the labels between 1-10.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Load the input batch
In_batch = load(filename);
 
%Number of labels
K = 10;

%Number of images 
data_size = size(In_batch.data,1);

%Matrix of images vectors (Normalized)
X = double(In_batch.data')/255;

%Lables changed from 0-9 to 1-10
y = In_batch.labels + 1;

%Inicializate the matrix of dimensions KxN
Y = zeros(K,data_size);

%Obtain the one-hot representation
for i = 1:K
    rows = y == i;
    Y(i, rows) = 1;
end
