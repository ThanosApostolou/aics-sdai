% This script demonstrates the classification performamce of a simple
% perceptron on a binary classification problem. Class1 (C1) training
% patterns are considered to be generated from a multivariate normal
% distribution with mean vector MEAN1 and covariance matrix SIGMA1. Class2
% (C2) training patterns, on the other hand, are considered to  
% be sampled from a multivariate normal distribution with mean vectot MEAN2 
% and covariance matrix SIGMA2.

%Initialize workspace.
clc
clear

% Define the parameters of the multivariate gaussian distributions.
MU1 = [4 4];
MU2 = [10 10];
SIGMA1 = [1 0; 0 1];
SIGMA2 = [1 0; 0 1];

% Define the number of training and testing patterns generated from both 
% classes.
NTrain = 100;
NTest = 100;
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
xlabel('x_1');
ylabel('x_2');
grid on
hold off

% Set the training patterns matrix for the perceptron object.
P = [C1Train;C2Train];
P = P';
% Set the target vector corresponding to the training patterns stored in P.
T = [zeros(1,NTrain),ones(1,NTrain)];
% Set the rank matrix for the perceptron object.
R = minmax(P);
% Set the perceptron object for the binary classification problem.
net = newp(R,1);
% Initialize Network.
net = init(net);
% Set perceptron training parameters.
net.trainParam.epochs = 350;
net.trainParam.goal = 0.0;
% Train perceptron.
[net,tr] = train(net,P,T);
% Plot the training performance of the network object.
% plotperform(tr);
% Get network predictions on training data.
YTrain = sim(net,P);
% Estimate the difference between predicted and actual labels.
DiffTrain = T - YTrain;
% Estimate the percentage of correctly classified training patterns.
CorrectTrainPercentage = length(find(DiffTrain==0)) / (2 * NTrain)

% Get the trained perceptron optimal weight vector (W) and bias term (B).
W = net.IW{1,1}
B = net.b{1}

% Estimate the boundary between the two classes.
x1 = [R(1,1):1:R(1,2)];
x2 = - (W(1)/W(2))*x1 - (B/W(2));

% Plot training data points and corresponding boundary.
% Plot the training data points to be fed within the perceptron.
figure('Name','Training Data Points and Boundary')
hold on
plot(C1Train(:,1),C1Train(:,2),'*r','LineWidth',1.4);
plot(C2Train(:,1),C2Train(:,2),'*g','LineWidth',1.4);
plot(x1,x2,'-k','LineWidth',1.5);
xlabel('x1');
ylabel('x2');
grid on
hold off