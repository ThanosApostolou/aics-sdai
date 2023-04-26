function [W] = BarabasiAlbert(No,M,N)

% This function implements the Barabasi-Albert algorithm for the generation
% of a random scale-free network utilizing a preferential attachment 
% mechanism.

Wcurrent = ones(No,No);
Idiag = [1:No+1:No*No];
Wcurrent(Idiag) = 0;
for n = No+1:1:N
    roulette = RouletteProbabilities(Wcurrent);
    Wnew = zeros(n,n);
    Wnew(1:n-1,1:n-1) = Wcurrent;
    for m = 1:1:M
        random_index = randi(length(roulette));
        selected_node_index = roulette(random_index);
        Wnew(selected_node_index,n) = 1;
        Wnew(n,selected_node_index) = 1;
        roulette = RouletteProbabilities(Wnew);
    end;
    Wcurrent = Wnew;
end;

W = Wcurrent;


end

