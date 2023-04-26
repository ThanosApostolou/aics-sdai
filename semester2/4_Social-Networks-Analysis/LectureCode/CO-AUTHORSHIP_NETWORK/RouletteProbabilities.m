function [P] = RouletteProbabilities(W)

n = size(W,2);
probabilities = sum(W,2) ./ sum(sum(W));
P = [];
for k = 1:1:n-1
    Nk = round(probabilities(k)*100);
    P = [P,k*ones(1,Nk)];
end;

end

