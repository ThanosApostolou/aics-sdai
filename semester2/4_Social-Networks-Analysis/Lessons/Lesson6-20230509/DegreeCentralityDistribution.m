function [H] = DegreeCentralityDistribution(Degrees)

% This function computes and displays the Degree Centrality distribution
% for a given vector of degree centralities.

min_degree = min(Degrees);
max_degree = max(Degrees);
degrees_range = [min_degree:max_degree];
H = hist(Degrees,degrees_range);
figure('Name','Degree Centrality Distribution');
bar(degrees_range,H);
axis([min_degree-1 max_degree+1 min(H) max(H)+5]);
xlabel('Degrees');
ylabel('Absolute Frequency');
grid on


end

