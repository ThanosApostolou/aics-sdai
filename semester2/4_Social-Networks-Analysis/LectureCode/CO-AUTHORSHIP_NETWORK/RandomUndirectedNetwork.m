function [W] = RandomUndirectedNetwork(N,P)

% This function generates the adjacency matrix of a random G(N,P) network
% where N is the number of nodes and P is the probability of each edge to
% emerge. W is the associated {0,1} adjacency matrix.

% Intitialize the adjacency matrix with uniform random entries within the
% [0,1] interval.
R = rand(N,N);
% All edges with Rij such that Rij <= P will be connected.
W = 1 - hardlim(R-P);
% Find the indices of the lower triangle matrix W.
Ilow = find(W==tril(W));
% Compute the transpose of W.
Wtrans = W';
% Assure that the adjacency matrix W is symmetric and therefore the graph
% in un-directed.
W(Ilow) = Wtrans(Ilow);
% Moreover, the diagonal elements of W must be set to zero.
Idiag = [1:N+1:N*N];
W(Idiag) = 0;

end

