pkg load statistics;
clc
clear
% Define the fundamental parameters
% of the two dimensional Gausian DIstribution


MU1 = [4 4];
NTrain = 100;
MU2 = [12 12];
NTest = 100;
SIGMA1 = eye(2);
SIGMA2 = eye(2);

% Generate the data sample from each class
C1Train = mvnrnd(MU1, SIGMA1, NTrain);
C2Train = mvnrnd(MU2, SIGMA2, NTrain);
C1Test = mvnrnd(MU1, SIGMA1, NTest);
C2Test = mvnrnd(MU2, SIGMA2, NTest);

figure('Name', 'Training Data Points')
hold on;
plot(C1Train(:,1), C1Train(:,2), '*r', 'lineWidth', 1.4)
plot(C2Train(:,1), C2Train(:,2), '*g', 'lineWidth', 1.4)
grid on;
xlabel('X_1');
ylabel('Y_1');
hold off;















