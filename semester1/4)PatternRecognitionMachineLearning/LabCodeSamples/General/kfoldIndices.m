function [TrainIndices,TestIndices] = kfoldIndices(N,K)
Indices = [1:1:N];
M = N / K;
if (mod(N,K)~=0)
    error('The number of elements within vector Indices must be fully devided by K');
else
    TrainIndices = cell(1,K);
    TestIndices = cell(1,K);
    for k = 1:1:K
        test_indices = [(k-1)*M+1:1:k*M]
        train_indices = setdiff(Indices,test_indices);
        TrainIndices{k} = train_indices;
        TestIndices{k} = test_indices;
    end;
end
