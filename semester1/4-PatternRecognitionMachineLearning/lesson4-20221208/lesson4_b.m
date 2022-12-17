% pkg load statistics;
clc
clear
% Define the fundamental parameters
% of the two dimensional Gausian DIstribution


MU1 = [2 2];
MU2 = [4 6];
MU3 = [6 2];
SIGMA1 = eye(2);
SIGMA2 = eye(2);
SIGMA3 = eye(2);
NTrain = 200;
NTest = 200;

C1Train = mvnrnd(MU1, SIGMA1, NTrain);
C2Train = mvnrnd(MU2, SIGMA2, NTrain);
C3Train = mvnrnd(MU3, SIGMA3, NTrain);

C1Test = mvnrnd(MU1, SIGMA1, NTest);
C2Test = mvnrnd(MU2, SIGMA2, NTest);
C3Test = mvnrnd(MU3, SIGMA3, NTest);

figure('Name', 'Training Data')
hold on;
scatter(C1Train(:,1), C1Train(:,2), 60, 'r', 'LineWidth', 2);
scatter(C2Train(:,1), C2Train(:,2), 60, 'g', 'LineWidth', 2);
scatter(C3Train(:,1), C3Train(:,2), 60, 'b', 'LineWidth', 2);
hold off;
grid on;
xlabel('x_1');
ylabel('x_2');






