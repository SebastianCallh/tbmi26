function [ labelsOut ] = kNN(X, k, Xt, Lt)
%KNN Your implementation of the kNN algorithm
%   Inputs:
%               X  - Features to be classified
%               k  - Number of neighbors
%               Xt - Training features
%               LT - Correct labels of each feature vector [1 2 ...]'
%
%   Output:
%               LabelsOut = Vector with the classified labels

% Make distance map

D = pdist2(X',Xt');
    
% The neighbours are in I
[~, I] = sort(D, 2);

% Take the mode for the K nearest neighbours
labelsOut = mode(Lt(I(:, 1:k)), 2);



