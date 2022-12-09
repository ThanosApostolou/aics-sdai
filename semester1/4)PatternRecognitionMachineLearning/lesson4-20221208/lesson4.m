% pkg load statistics;
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

% NEW CODE:
P = [C1Train; C2Train];
P = P';
T = [zeros(1, NTrain), ones(1, NTrain)];
R = minmax(P);
net = newp(R, 1);
net = init(net);
net.trainParam.epochs = 400;
net.trainParam.goal = 0.0;

[net, tr] = train(net, P, T);
% plotperform(tr);
YTrain = sim(net, P);
DiffTrain = T - YTrain;
CorrectTrainingPercentage = length(find(DiffTrain == 0));

W = net.IW{1,1};
B = net.b{1};

x1 = R(1,1):1:R(1,2);
% w1x1 + w2x2 + B = 0
% => x2 = -(w1/w2)x1 - B/w2
x2 = -(W(1) / W(2)) * x1 - (B / W(2));

figure('Name', 'Training Data oints and Bounding')
hold on;
plot(C1Train(:,1), C1Train(:,2), '*r', 'LineWidth', 1.4);
plot(C2Train(:,1), C2Train(:,2), '*g', 'LineWidth', 1.4);
plot(x1, x2, '-K', 'LineWidth', 1.5)
xlabel('x_1');
ylabel('x_2');
grid on;
hold off;

% import myfunc.*
% [x, y] = myfunc()




