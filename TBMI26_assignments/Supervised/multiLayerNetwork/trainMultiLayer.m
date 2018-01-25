function [Wout,Vout, trainingError, testError ] = trainMultiLayer(Xtraining,Dtraining,Xtest,Dtest, W0, V0,numIterations, learningRate )
%TRAINMULTILAYER Trains the network (Learning)
%   Inputs:
%               X* - Trainin/test features (matrix)
%               D* - Training/test desired output of net (matrix)
%               V0 - Weights of the output neurons (matrix)
%               W0 - Weights of the output neurons (matrix)
%               numIterations - Number of learning setps (scalar)
%               learningRate - The learningrate (scalar)
%
%   Output:
%               Wout - Weights after training (matrix)
%               Vout - Weights after training (matrix)
%               trainingError - The training error for each iteration
%                               (vector)
%               testError - The test error for each iteration
%                               (vector)

% Initiate variables
trainingError = nan(numIterations+1,1);
testError = nan(numIterations+1,1);
numTraining = size(Xtraining,2);
numTest = size(Xtest,2);
numClasses = size(Dtraining,1) - 1;
Wout = W0;
Vout = V0;
Nt = size(Xtraining,2);

% Calculate initial error
Ytraining = runMultiLayer(Xtraining, W0, V0);
Ytest = runMultiLayer(Xtest, W0, V0);
trainingError(1) = sum(sum((Ytraining - Dtraining).^2))/(numTraining*numClasses);
testError(1) = sum(sum((Ytest - Dtest).^2))/(numTest*numClasses);
grad_v = 0;
grad_w = 0;
momentum = 0.8;

for n = 1:numIterations
    [Ytraining,~, U] = runMultiLayer(Xtraining, Wout, Vout);

    grad_v_old = learningRate * grad_v;
    grad_w_old = learningRate * grad_w;
    grad_v = 2/Nt*(Ytraining-Dtraining)*U'; %Calculate the gradient for the output layer
    grad_w = 2/Nt*(Vout'*(Ytraining-Dtraining).*(1 - U.^2))*Xtraining'; %..and for the hidden layer.
    grad_w = grad_w(2:end,:);
    if mod(n,1000) == 0
        disp(n)
        disp(testError(n))
    end
    %size(grad_w)
    %size(Wout)
    Wout = Wout - (learningRate * grad_w + momentum*grad_w_old); %Take the learning step.
    Vout = Vout - (learningRate * grad_v + momentum*grad_v_old); %Take the learning step.

    Ytraining = runMultiLayer(Xtraining, Wout, Vout);
    Ytest = runMultiLayer(Xtest, Wout, Vout);
    
    trainingError(1+n) = sum(sum((Ytraining - Dtraining).^2))/(numTraining*numClasses);
    testError(1+n) = sum(sum((Ytest - Dtest).^2))/(numTest*numClasses);
end

end
