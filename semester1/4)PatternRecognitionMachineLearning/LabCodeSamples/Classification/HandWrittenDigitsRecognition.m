% Clear all existing variables.
clear all
% Load hand written digits images from file usps_all.mat in variable data.
load('usps_all.mat');
% Matrix data is 256 x 1100 x 10 , 3 - dimensional array storing a number
% of 1100 (16 x 16 pixels) images for each digit.
% Keep only those images corresponding to the hand written versions of 0
% and 1.
X = [data(:,:,1)';data(:,:,10)'];
X = double(X);
X = X ./ 255;
Y = hist(X',5);

plot3(Y(1,[1:1:1100]),Y(2,[1:1:1100]),Y(3,[1:1:1100]),'*r')
hold on
plot3(Y(1,[1101:1:2200]),Y(2,[1101:1:2200]),Y(3,[1101:1:2200]),'*g')
grid on
hold off