function [W] = WattsStrogatz(N,K,beta)

% This function implements the Watts-Strogatz algorithm for the generation
% of a random graph with small-world properties.
% N is the number of network nodes to be generated.
% K is the mean degree which is assummed to be an even integer.
% beta is a special parameter assumming values withing the [0,1] interval.
% Morover, the following inequality must hold:
% N >> K >> ln(N) >> 1.

% Construct a regular ring lattice.
W = zeros(N,N);
for n = 0:1:N-1
    for k = 1:1:(K/2)
        right_edge_node = mod((n+k),N);
        left_edge_node = mod((n-k),N);
        W(n+1,right_edge_node+1) = 1;
        W(n+1,left_edge_node+1) = 1;
    end;
end;
% Rewire existing nodes.
for i = 1:1:N
    for j = i+1:1:N
        if(W(i,j)==1)
            R = rand();
            if(R<=beta)
                possible_nodes = [1:1:N];
                connected_nodes = find(W(i,:)==1);
                self_node = i;
                valid_nodes = setdiff(possible_nodes,connected_nodes);
                valid_nodes = setdiff(valid_nodes,self_node);
                random_index = randi(length(valid_nodes));
                k = valid_nodes(random_index);
                W(i,j) = 0;
                W(j,i) = 0;
                W(i,k) = 1;
                W(k,i) = 1;
            end;
        end;
    end;
end;


end

