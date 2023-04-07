% Να υπολογιστεί το EigenVector centrality για κάθε κόμβο του παρακάτω γραφήματος
% Γράφο σχήματος Cayley tree
clc
clear


No = 10;
N = cell(1, No);
N{1} = [2, 3, 4];
N{2} = [1, 5, 6];
N{3} = [1, 7, 8];
N{4} = [1, 9, 10];
N{5} = 2;
N{6} = 2;
N{7} = 3;
N{8} = 3;
N{9} = 4;
N{10} = 4;


W = zeros(No, No);
for k = 1:1:No
    W(k, N{k}) = 1;
end
G = graph(W);
plot(G);

[V, D] = eig(W);

max_eigenvalue = max(max(D));
[max_i, max_j] = find(D==max_eigenvalue);
EC = V(:, max_j);
EC = EC/sum(EC);
