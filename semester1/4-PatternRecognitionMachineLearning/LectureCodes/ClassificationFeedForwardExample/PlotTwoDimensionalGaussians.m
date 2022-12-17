function PlotTwoDimensionalGaussians(R1,R2,MU1,MU2,SIGMA1,SIGMA2)

% This function provides a three-dimensional plot for a pair of  
% two-dimensional gaussian probability density functions which ara
% parameterized by MU1, SIGMA1 and MU2, SIGMA2 respectively.
% The general form of a two-dimensional gausian pdf with mean vector MU and
% covariance matrix SIGMA is given by the following equation:
%                  1                  1
% P(x1,x2) = -------------- x exp{ - --- [x1'^2*S(1,1) + (S(1,2)+S(2,1))*x1'*x2' + x2'^2*S(2,2)]}
%            2*pi*det(SIGMA)          2
% where:
% x1' = x1 - MU_1 and x2' = x2 - MU_2 given that MU = [MU_1 MU_2] and
% S = inv(SIGMA).
% R1 and R2 are [2x2] matrices of the following forms:
%      |x11_min x11_max|          |x21_min x22_max|
% R1 = |x12_min x12_max| and R2 = |x22_min x22_max|

% Set the interval step.
d_x1_x2 = 0.01;
% Construct the free parameter vectors x11, x12 and x21, x22.
x11 = R1(1,1):d_x1_x2:R1(1,2);
x12 = R1(2,1):d_x1_x2:R1(2,2);
x21 = R2(1,1):d_x1_x2:R2(1,2);
x22 = R2(2,1):d_x1_x2:R2(2,2);
% Construct two meshgrids for the free parameter vectors.
% The meshgrid for the first pdf will be denoted as [X11 x X12], while the
% meshgrid for the second pdf will be denoted as [X21 x X22].
[X11,X12] = meshgrid(x11,x12);
[X21,X22] = meshgrid(x21,x22);
% Get the corresponding prime vector forms X11p, X12p and X21p, X22p.
X11p = X11 - MU1(1);
X12p = X12 - MU1(2);
X21p = X21 - MU2(1);
X22p = X22 - MU2(2);
% Get the inverse covariance matrices SIGMA1inv and SIGMA2inv.
SIGMA1inv = inv(SIGMA1);
SIGMA2inv = inv(SIGMA2);
% Compute the quantities EXP_FACTOR1 and EXP_FACTOR2 that appear within the
% exponentials.
EXP_FACTOR1 = X11p.^2 * SIGMA1inv(1,1) + ...
              (SIGMA1inv(1,2)+SIGMA1inv(2,1)) * (X11p .* X12p) + ...
              X12p.^2 * SIGMA1inv(2,2);
EXP_FACTOR2 = X21p.^2 * SIGMA2inv(1,1) + ...
              (SIGMA2inv(1,2)+SIGMA2inv(2,1)) * (X21p .* X22p) + ...
              X22p.^2 * SIGMA2inv(2,2);
% Compute the pdf quantities P1 and P2.
P1 = exp(-0.5 * EXP_FACTOR1);
P1 = P1 * (1/(2*pi*sqrt(det(SIGMA1))));
P2 = exp(-0.5 * EXP_FACTOR2);
P2 = P2 * (1/(2*pi*sqrt(det(SIGMA2))));

% Plot pdfs.
figure('Name','Two-Dimensional Gaussian PDFs');
% Red and Blue Colormap.
colormap([1 0 0;0 0 1])
surf(X11,X12,P1);
hold on
surf(X21,X22,P2);
hold off
xlabel('Feature_1');
ylabel('Feature_2');
zlabel('Probability');

% Plot surfaces.
figure('Name','Two-Dimensional Gaussian Surfaces');
hold on
contour(X11,X12,P1);
contour(X21,X22,P2);
hold off
grid on
xlabel('Feature_1');
ylabel('Feature_2');

end

