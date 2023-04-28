% Clear all existing variables.
clc
clear all
% Load hand written digits images from file usps_all.mat in variable data.
load('usps_all.mat');
% Matrix data is 256 x 1100 x 10 , 3 - dimensional array storing a number
% of 1100 (16 x 16 pixels) images for each digit.
% Keep only those images corresponding to the hand written versions of 0
% and 1.
ZerosImages = data(:,:,10)';
OnesImages = data(:,:,1)';
X = [ZerosImages;OnesImages];
% Thus, the data matrix X must be a 2200 x 256 matrix where each row
% corresponds to the pixel values of 16 x 16 image. The first 1100 rows
% store images of ones while the last 1100 rows correspond to images
% store images of zeros.

% Matrix X is a uint8 array that must be converted to a double array.
X = double(X);
% Normalize the pixel values in the [0 1] interval.
X = X ./ 255;
% Feature Selection Process.
% Set as feature the average intensity level per image.
Y = mean(X,2);
% Set the training and test patterns.
% Keep the first 1000 images from each class for training the Feedforward
% Neural Network and test with the remaining 100 images from each class.
TrainingPatterns = Y([1:1:1000,1101:1:2100]);
TestingPatterns = Y([1001:1:1100,2101:1:2200]);

NTrain = length(TrainingPatterns)/2;
NTest = length(TestingPatterns)/2;
% Get the training feature vectors per class.
P1x = Y([1:1:1000]);
P2x = Y([1101:1:2100]);
% Check the discrimination efficiency of the utilized feature, namely the
% average intensity level.
% Get the minimum and maximum values for the utilized feature per class.
P1x_min = min(P1x);
P1x_max = max(P1x);
P2x_min = min(P2x);
P2x_max = max(P2x);
% Set internal plotting parameters the class conditional probality density 
% functions per class.
step_percentage = 0.01;
P1x_step = (P1x_max - P1x_min) * step_percentage;
P1x_interval = [P1x_min:P1x_step:P1x_max];
P2x_step = (P2x_max - P2x_min) * step_percentage;
P2x_interval = [P2x_min:P2x_step:P2x_max];
% Fit a normal distribution per class for the utilized feature.
[P1x_mu,P1x_sigma] = normfit(P1x);
[P2x_mu,P2x_sigma] = normfit(P2x);
% Get the class conditional probability density functions for the utilized
% feature.
P1x_norm_pdf = normpdf(P1x_interval,P1x_mu,P1x_sigma);
P2x_norm_pdf = normpdf(P2x_interval,P2x_mu,P2x_sigma);
% Get the class conditional probability density functions for the utilized
% feature.
figure('Name','Class Conditional Probability Density Functions');
hold on
plot(P1x_interval,P1x_norm_pdf,'-r','LineWidth',1.6);
plot(P2x_interval,P2x_norm_pdf,'-g','LineWidth',1.6);
xlabel('Average Intensity Level');
ylabel('Probability Mass');
grid on
hold off

% Set the training patterns matrix for the perceptron object.
P = TrainingPatterns;
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
net.trainParam.epochs = 1000;
net.trainParam.goal = 0.0;
net.trainFcn = 'trainb';
% Train perceptron.
[net,tr] = train(net,P,T);
%Plot the training performance of the network object.
figure()
plotperform(tr);
% Get network predictions on training data.
YTrain = sim(net,P);
% Estimate the difference between predicted and actual labels.
DiffTrain = abs(T - YTrain);
% Estimate the percentage of correctly classified training patterns.
CorrectTrainPercentage = length(find(DiffTrain==0)) / (2 * NTrain)
% Get the trained perceptron optimal weight vector (W) and bias term (B).
W = net.IW{1,1};
B = net.b{1};

% Get the desiction threhold.
x_thres = - (B / W)

% Set the testing patterns matrix for the perceptron object.
P = TestingPatterns;
P = P';
% Set the target vector corresponding to the training patterns stored in P.
T = [zeros(1,NTest),ones(1,NTest)];
% Get network predictions on training data.
YTest = sim(net,P);
% Estimate the difference between predicted and actual labels.
DiffTest = abs(T - YTest);
% Estimate the percentage of correctly classified training patterns.
CorrectTestPercentage = length(find(DiffTest==0)) / (2 * NTest)