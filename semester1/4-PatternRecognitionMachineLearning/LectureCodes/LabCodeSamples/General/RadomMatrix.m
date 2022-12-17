clc
clear all
% Set the elements of the random matrix A that are uniformly distributed in
% the [1..10] interval.
A = ceil(10*rand(100,100));
EvenRows = A([2:2:100],:);
EvenRowsMean = sum(sum(EvenRows))/numel(EvenRows);
% or equivalently
EvenRowsMean = mean(mean(EvenRows));
OddColumns = A(:,[1:2:99]);
OddColumnsMean = mean(mean(OddColumns));
% or equivalently
OddColumnsMean = sum(sum(OddColumns))/numel(OddColumns);
% Transform matrix A into a row vector.
V = reshape(A,1,numel(A));
% Compute the histogram of frequencies.
F = hist(V,[1:1:10]);
figure('Name','Frequency Histogram')
bar([1:1:10],F,'b')
xlabel('Integer Values')
ylabel('Absolute Frequencies')
grid on
% Compute the mean frequencies of even and odd elements in the [1..10] elements.
Feven_mean = mean(F([2:2:10]))
Fodd_mean = mean(F([1:2:9]))