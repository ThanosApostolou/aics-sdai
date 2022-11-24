% This script demonstrates the classification performamce of a simple
% linear network on a three class classification problem by considering 
% two dimensional feature vectors. Patterns pertaining to ClassX where 
% X in {1,2,3} are considered to be sampled from a multivariate normal
% distribution wih mean vector MUX and covariance matrix SIGMAX.

% Clear command window and workspace.
clc
clear

% Define the parameters of the multivariate gaussian distributions such
% that the constituent values of the corresponding feature vectors are 
% uncorrelated.
MU1 = [2 2];
MU2 = [4 6];
MU3 = [6 2];
SIGMA1 = eye(2);
SIGMA2 = eye(2);
SIGMA3 = eye(2);

% Define the number of training and testing patterns generated for each 
% class.
NTrain = 200;
NTest = 200;

% Generate training patterns for each one of the classes C1, C2 and C3.
C1Train = mvnrnd(MU1,SIGMA1,NTrain);
C2Train = mvnrnd(MU2,SIGMA2,NTrain);
C3Train = mvnrnd(MU3,SIGMA3,NTrain);
% Generate testing patterns for each one of the classes C1, C2 and C3.
C1Test = mvnrnd(MU1,SIGMA1,NTest);
C2Test = mvnrnd(MU2,SIGMA2,NTest);
C3Test = mvnrnd(MU3,SIGMA3,NTest);

% Plot the training data points to be fed in the linear neural network.
figure('Name','Training Data Points')
hold on
scatter(C1Train(:,1),C1Train(:,2),60,'r','LineWidth',2.0);
scatter(C2Train(:,1),C2Train(:,2),60,'g','LineWidth',2.0);
scatter(C3Train(:,1),C3Train(:,2),60,'b','LineWidth',2.0);
hold off
grid on
xlabel('x_1');
ylabel('x_2');

% Set the matrix of training patterns for the linear neural network.
Ptrain = [C1Train;C2Train;C3Train]';
% Set the matrix of training targets for the linear neural network.
Ttrain = [ones(1,NTrain),2*ones(1,NTrain),3*ones(1,NTrain)];
Ttrain = full(ind2vec(Ttrain));

% Set up a linear neural network and perform the following operations:
% (i)  :  network configuration according to the corresponding set of training patterns.
% (ii) :  network initilization.
% (iii):  network data division function initialization.
% (iv) :  network training function determination.
% (v)  :  enable verval output during training.
% (vi) :  enable visual output during training.
linear_net = linearlayer;
linear_net = configure(linear_net,Ptrain,Ttrain);
linear_net = init(linear_net);
linear_net.divideFcn = '';
linear_net.trainFcn = 'trainrp';
linear_net.trainParam.showCommandLine = true;
linear_net.trainParam.showWindow = true;
linear_net.trainParam.epochs = 1000;
linear_net.trainParam.goal = 0.0;

% View the linear network object.
view(linear_net);

% Train the linear neural network.
linear_net = train(linear_net,Ptrain,Ttrain);

% Simulate the linear neural network response on the training data.
Ttrain_est = linear_net(Ptrain);
% Convert the estimated output vector into the corresponding class labels
% by storing the index of the maximum value per column of the output
% matrix.
[~,Ttrain_est] = max(Ttrain_est);

% Compute the confussion matrix for training.
CMtrain = confusionmat(vec2ind(Ttrain),Ttrain_est);
% Visualize the confussion matrix for training.
confusionchart(CMtrain);