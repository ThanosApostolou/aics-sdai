clc
clear all
N = 10;
% Check what happens for N = 100, N = 1000 and N = 5000.
% Which routine is faster and why?
% Version 1 - Non Vectorized Code.
tic
S = 0;
for m = 1:1:N
    for n = 1:1:N
        S = S + m^2 + n^2;
    end;
end;
toc
S
% Version 2 - Vectorized Code.
tic
S = 0;
I = [1:1:N]' * ones(1,N);
W = I.^2 + (I').^2;
S = sum(sum(W));
toc
S
% Version 3 - Vectorized Code.
tic 
S = 0;
I = repmat([1:1:N]',1,N);
W = I.^2 + (I').^2;
S = sum(sum(W));
toc
S