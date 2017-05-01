function [ X, Y, y ] = LoadBatch2( filename )
%LOADBATCH Obtains the pixel values, label and one-hot-encoded label for a
%batch of images
%   [X, Y, y ] = LoadBatch( filename )
%
% Inputs:
%   filename: Name of the file to load

A = load(filename);

% Pixel data, size dxN (d: #features, N: #samples)
X = double(A.data)'/255;

% Label data, size N
y = double(A.labels+1);

% One-hot encoding of the label data, size KxN (K: #Classes)
Y = full(ind2vec(y'));
end