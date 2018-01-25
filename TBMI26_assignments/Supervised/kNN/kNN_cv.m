function [ perfs ] = kNN_cv(X, D, L)
%KNN Your implementation of the kNN algorithm
%   Inputs:
%               X  - Features to be classified
%               k  - Number of neighbors
%               Xt - Training features
%               LT - Correct labels of each feature vector [1 2 ...]'
%
%   Output:
%               LabelsOut = Vector with the classified labels


numBins = 5; % Number of Bins you want to devide your data into
numSamplesPerLabelPerBin = 20; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true; % true = select features at random, false = select the first features

[ Xt, ~, Lt ] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom );

ks = 25;
perfs = zeros(ks, 1);
for k = 1:ks 
    
    % Array for storing performance
    accs = 0;
    for n = 1:numBins
    
        % Training data and labels
        T = [];
        L = [];
        for i = 1:numBins
           if (i ~= n) 
            T = [T  Xt{i}];
            L = [L; Lt{i}];
           end
        end
        
        
        LkNN = kNN(Xt{n}, k, T, L); 
        cM   = calcConfusionMatrix(LkNN, Lt{n});
        accs = accs + calcAccuracy(cM);    
    end
    perfs(k) = accs / numBins;
end