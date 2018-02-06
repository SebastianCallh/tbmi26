% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces);
nonfaces = double(nonfaces);

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

% Generate Haar feature masks
nbrHaarFeatures = 25;
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

% Create a training data set with a number of training data examples
% from each class. Non-faces = class label y=-1, faces = class label y=1
nbrTrainExamples = 50;
trainImages = cat(3,faces(:,:,1:nbrTrainExamples),nonfaces(:,:,1:nbrTrainExamples));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainExamples), -ones(1,nbrTrainExamples)];

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError
n = size(xTrain, 2);
d = ones(1, n) / n;
m = 50;
ts = zeros(m, 1);
ps = zeros(m, 1);
ks = zeros(m, 1);
as = zeros(m, 1);

%%
for i = 1:m
    e_min = inf;
    t_min = 0;
    p_min = 0;
    k_min = 0;
    a_min = 0;
    disp('New Classifier')
    for k = 1:size(xTrain, 1)
        thresholds = xTrain(k,:) + 0.01;
        disp('New Feature')
        for t = thresholds
            p   = 1;
            h   = WeakClassifier(t, p, xTrain(k,:));
            e_t = WeakClassifierError(h, d, yTrain);

            if e_t > 0.5 
                p = -p;
                e_t = 1 - e_t;
            end

            if e_t < e_min
                e_min = e_t
                a = log((1 - e_min) / e_min) / 2;
                t_min = t;
                p_min = p;
                k_min = k;
                a_min = a;
            end
        end    
    end
    
    if e_min == 0.5
        break;
    end
        
    d = exp(-a * yTrain .* h);
    d = d ./ sum(d);
    ts(i) = t_min;
    ps(i) = p_min;
    ks(i) = k_min;
    as(i) = a_min;
end
%%
k = 25;
e_min = inf
for t = thresholds
    p   = 1;
    h   = WeakClassifier(t, p, xTrain(k,:));
    e_t = WeakClassifierError(h, d, yTrain);
    
    if e_t > 0.5
        p = -p;
        e_t = 1 - e_t;
    end
    
    if e_t < e_min
        e_min = e_t
        a = log((1 - e_min) / e_min) / 2;
        h
        yTrain
        t_min = t
        p_min = p;
        k_min = k;
        a_min = a;
    end
end
%% Extract test data

nbrTestExamples = 1000;
testImages  = cat(3,faces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)),...
                    nonfaces(:,:,(nbrTrainExamples+1):(nbrTrainExamples+nbrTestExamples)));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,nbrTestExamples), -ones(1,nbrTestExamples)];

%% Evaluate your strong classifier here
%  You can evaluate on the training data if you want, but you CANNOT use
%  this as a performance metric since it is biased. You MUST use the test
%  data to truly evaluate the strong classifier.

cs = zeros(length(ts), n);
for i = 1:length(cs)
    i
    cs(i,:) = as(i) .* WeakClassifier(ts(i), ps(i), yTrain(ks(i),:));
end
 
sign(sum(cs))

%% Plot the error of the strong classifier as  function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.


