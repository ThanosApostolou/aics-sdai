% This script demonstrates the hierarchical clustering algorithm on a set of
% artificially generated data points. Data points are considered to be
% sampled from three different multivariate distributions identified by the
% parameters MU1, MU2, MU3 and SIGMA1, SIGMA2, SIGMA3.

%Initialize workspace.
clc
clear all

% Define the parameters of the multivariate gaussian distributions.
MU1 = [4 4];
MU2 = [6 6];
MU3 = [8 8];
SIGMA1 = [1 0; 0 1];
SIGMA2 = [1 0; 0 1];
SIGMA3 = [1 0; 0 1];

% Sample equal number of points from both distributions.
% Let N be the number of points to be sampled.
N = 200;
X1 = mvnrnd(MU1,SIGMA1,N);
X2 = mvnrnd(MU2,SIGMA2,N);
X3 = mvnrnd(MU3,SIGMA3,N);
% Store both sets of points in a single matrix.
X = [X1;X2;X3];

% Plot the labeded data points
figure('Name','Labeled Data Popints')
hold on
plot(X1(:,1),X1(:,2),'*r','LineWidth',1.4);
plot(X2(:,1),X2(:,2),'*b','LineWidth',1.4);
plot(X3(:,1),X3(:,2),'*g','LineWidth',1.4);
xlabel('x1');
ylabel('x2');
grid on
hold off

%Plot the unlabaled data points.
figure('Name','Unlabeled Data Popints')
hold on
plot(X(:,1),X(:,2),'*k','LineWidth',1.4);
xlabel('x1');
ylabel('x2');
grid on
hold off

% Create the hierarchical clustering dendrogram.
Y = pdist(X,'euclidean');
Z = linkage(Y,'average');
figure('Name','Hierarchical Dendrogram')
[H,T] = dendrogram(Z,0,'colorthreshold','default');
set(H,'LineWidth',2)

% Cluster data in three clusters.
T = clusterdata(X,'linkage','average',3);
figure('Name','Identified Clusters')
scatter(X(:,1),X(:,2),10,T,'filled')
grid on