% This script demonstrates the classification performamce of feedforward
% neural network  on a binary classification problem. Class1 (C1) training
% patterns are considered to be generated from a multivariate normal
% distribution with mean vector MEAN1 and covariance matrix SIGMA1. Class2
% (C2) training patterns, on the other hand, are considered to  
% be sampled from a multivariate normal distribution with mean vectot MEAN2 
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

% Plot the propability density function for the class patterns of each
% class.
P1 = C1Train;
P2 = C2Train;
% Get the range matrices for each class of patterns.
R1 = minmax(P1');
R2 = minmax(P2');
% Treat seperately the x and y constituents for the training vectors of
% each class.
P1x = P1(:,1);
P1y = P1(:,2);
P2x = P2(:,1);
P2y = P2(:,2);
% Get the minimum and maximum values of the x and y constituents for the
% training vectors of each class.
P1x_min = R1(1,1);
P1x_max = R1(1,2);
P1y_min = R1(2,1);
P1y_max = R1(2,2);

P2x_min = R2(1,1);
P2x_max = R2(1,2);
P2y_min = R2(2,1);
P2y_max = R2(2,2);

% Treat seperately the parameters of the two dimensional gaussian  
% probability density function for each dimension.
MU1x = MU1(1);
MU1y = MU1(2);
MU2x = MU2(1);
MU2y = MU2(2);

% Store the inverse covariance matrices corresponding to the class
% conditional probality density functions for each class of patterns.
SIGMA1inv = inv(SIGMA1);
SIGMA2inv = inv(SIGMA2);

% Get the x and y axis intervals corresponding to the 2 dimensional 
% features of each class.
dxy = 0.1;
x1 = [P1x_min:dxy:P1x_max];
y1 = [P1y_min:dxy:P1y_max];
x2 = [P2x_min:dxy:P2x_max];
y2 = [P2y_min:dxy:P2y_max];

% Get the required meshgrids for plotting in 3 dimensions for both
% gaussians.
[X1 Y1] = meshgrid(x1,y1);
[X2 Y2] = meshgrid(x2,y2);

% Get the probability density values for each class of patterns at the
% corresponding mesgrid points.
Prob1 = 1 / (2*pi*sqrt(det(SIGMA1))) * exp(-0.5*(SIGMA1inv(1,1)*(X1-MU1x).^2 + ...
                                        (SIGMA1inv(1,2)+SIGMA1inv(2,1))*(X1-MU1x).*(Y1-MU1y) + ...
                                         SIGMA1inv(2,2)*(Y1-MU1y).^2));

Prob2 = 1 / (2*pi*sqrt(det(SIGMA2))) * exp(-0.5*(SIGMA2inv(1,1)*(X2-MU2x).^2 + ...
                                        (SIGMA2inv(1,2)+SIGMA2inv(2,1))*(X2-MU2x).*(Y2-MU2y) + ...
                                         SIGMA1inv(2,2)*(Y2-MU2y).^2));                                     
                                     
% Plot the corresponding gaussian density functions.
figure('Name','Probability Density Functions');
hold on
surf(x1,y1,Prob1);
surf(x2,y2,Prob2);
hold off
% axis vis3d
% camorbit(45,45,'data',[1 0 1])
% drawnow

% Plot the corresponding gaussian density functions contours.
figure('Name','Probability Density Functions Contours');
hold on
contour(x1,y1,Prob1);
contour(x2,y2,Prob2);
hold off

% Set the training patterns matrix for the feed forward neural network object.
P = [C1Train;C2Train];
P = P';
% Set the target vector corresponding to the training patterns stored in P.
T = [ones(1,NTrain),2*ones(1,NTrain)];
%T = full(ind2vec(T));

% This code segment happens to be obsolete in Matlab 2010b version and later.
% % Set the range matrix for the feedforard neural network object.
% R = minmax(P);

% Set the neural network
net = newff(P,T,[8 4 4 1],{'tansig' 'tansig' 'tansig' 'purelin'});
init(net);
net.trainParam.epochs = 500;
net.trainParam.showCommandLine = 1;
%net.trainParam.goal = 0.01;
net = train(net,P,T);

% Check network performance on training patterns.
EstimatedTrainingTargets = sim(net,P);
EstimatedTrainingTargets = round(EstimatedTrainingTargets);
Differences = abs(EstimatedTrainingTargets - T);
CorrectClassificationRatio = 1 - (sum(sum(Differences)) / (2*NTrain))