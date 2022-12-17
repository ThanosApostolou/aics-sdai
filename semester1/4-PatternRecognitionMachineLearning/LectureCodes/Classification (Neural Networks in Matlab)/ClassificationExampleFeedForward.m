% This script demonstrates the classification performamce of feedforward
% neural network  on a binary classification problem. Class1 (C1) training
% patterns are considered to be generated from a multivariate normal
% distribution with mean vector MEAN1 and covariance matrix SIGMA1. Class2
% (C2) training patterns, on the other hand, are considered to  
% be sampled from a multivariate normal distribution with mean vector MEAN2 
% and covariance matrix SIGMA2.

%Initialize workspace.
clc
clear all

% Define the parameters of the multivariate gaussian distributions.
MU1 = [2 2];
MU2 = [3 3];
SIGMA1 = [1 0; 0 1];
SIGMA2 = [1 0; 0 1];

% Define the number of training and testing patterns generated from both 
% classes.
NTrain =200;
NTest = 200;
% Generate training patterns from classes Class1 and Class2 (C1 and C2).
C1Train = mvnrnd(MU1,SIGMA1,NTrain);
C2Train = mvnrnd(MU2,SIGMA2,NTrain);
% Generate testing patterns from classes Class1 and Class2 (C1 and C2).
C1Test = mvnrnd(MU1,SIGMA1,NTest);
C2Test = mvnrnd(MU2,SIGMA2,NTest);

% Plot the training data points to be fed within the perceptron.
figure('Name','Training Data Points')
hold on
plot(C1Train(:,1),C1Train(:,2),'*r','LineWidth',1.4);
plot(C2Train(:,1),C2Train(:,2),'*g','LineWidth',1.4);
xlabel('x1');
ylabel('x2');
legend('Class1','Class2');
grid on
hold off

% Get the range matrices for each class of patterns.
R1 = minmax(C1Train');
R2 = minmax(C2Train');
% Plot the two-dimensional gaussians.
PlotTwoDimensionalGaussians(R1,R2,MU1,MU2,SIGMA1,SIGMA2);

% Set the training patterns matrix for the feed forward neural network object.
P = [C1Train;C2Train];
P = P';
% Set the target vector corresponding to the training patterns stored in P.
T = [ones(1,NTrain),2*ones(1,NTrain)];

% Set the neural network
net = newff(P,T,[4 2 2],{'tansig' 'tansig' 'purelin'});
init(net);
net.trainParam.epochs = 2000;
net.trainParam.showCommandLine = 1;
net.trainParam.goal = 0.00001;
net.trainParam.lr = 0.001;
net.trainFcn = 'trainbr';
net = train(net,P,T);

% Check network performance on training patterns.
EstimatedTrainingTargets = sim(net,P);
EstimatedTrainingTargets = round(EstimatedTrainingTargets);
Differences = abs(EstimatedTrainingTargets - T);
CorrectTrainClassificationRatio = 1 - (sum(Differences) / (2*NTrain));

% Plot decision boundaries.
PlotDecisionBoundaries(R1,R2,net);

% Set the testing patterns matrix for the feed forward neural network object.
P = [C1Test;C2Test];
P = P';
% Set the target vector corresponding to the training patterns stored in P.
T = [ones(1,NTest),2*ones(1,NTest)];

% Check network performance on testing patterns.
EstimatedTestingTargets = sim(net,P);
EstimatedTestingTargets = round(EstimatedTestingTargets);
Differences = abs(EstimatedTestingTargets - T);
CorrectTestClassificationRatio = 1 - (sum(sum(Differences)) / (2*NTest))