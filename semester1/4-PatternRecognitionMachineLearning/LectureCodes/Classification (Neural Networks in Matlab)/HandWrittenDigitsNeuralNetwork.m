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
H = hist(X',5);
% Transpose matrix H so that each row contains the histogram of every
% image.
H = H';

% Set the training and test patterns.
% Keep the first 1000 images from each class for training the Feedforward
% Neural Network and test with the remaining 100 images from each class.

TrainingPatterns = H([1:1:1000,1101:1:2100],:);
TestingPatterns = H([1001:1:1100,2101:1:2200],:);

TrainingPatternsNum = size(TrainingPatterns,1);
TestingPatternsNum = size(TestingPatterns,1);

% This code segment happens to be obsolete in Matlab 2010b version and later.
% Set the Range matrix needed for the construction of the FFNN.
%Range = minmax(TrainingPatterns')

% Set the training targets.
TrainingTargets = zeros(1,TrainingPatternsNum);
TrainingTargets([1:1:1000]) = 1;
TrainingTargets([1001:1:2000]) = 2;
TrainingTargets= full(ind2vec(TrainingTargets));

% Set the testing targets.
TestingTargets = zeros(1,TestingPatternsNum);
TestingTargets([1:1:100]) = 1;
TestingTargets([101:1:200]) = 2;
TestingTargets= full(ind2vec(TestingTargets));

% Transpose TrainingPatterns and Testting Patterns matrix so that each training pattern
% corresponds to a different column.
TrainingPatterns = TrainingPatterns';
TestingPatterns = TestingPatterns';

% Set the neural network

net = newff(TrainingPatterns,TrainingTargets,[10 6 2],{'tansig' 'tansig' 'purelin'});
init(net);
net.trainParam.epochs = 80;
net.trainParam.showCommandLine = 1;
net = train(net,TrainingPatterns,TrainingTargets);

% Check network performance on training patterns.

EstimatedTrainingTargets = sim(net,TrainingPatterns);
EstimatedTrainingTargets = round(EstimatedTrainingTargets);
EstimatedTrainingTargetsVector = and(EstimatedTrainingTargets,TrainingTargets); % Get the correct classifications
CorrectTrainClassificationRatio = sum(sum(EstimatedTrainingTargetsVector)) / TrainingPatternsNum

% Check network performance on testing patterns.

EstimatedTestingTargets = sim(net,TestingPatterns);
EstimatedTestingTargets = round(EstimatedTestingTargets);
EstimatedTestingTargetsVector = and(EstimatedTestingTargets,TestingTargets); % Get the correct classifications
CorrectTestClassificationRatio = sum(sum(EstimatedTestingTargetsVector)) / TestingPatternsNum