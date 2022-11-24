% Clear screen.
clc
% Clear all variables in the working space.
clear all
% Set the dimensionality of the square matrix.
n = 3;
% Compute the number of elements in the corresponding square matrix.
N = n * n;
% Set the elements of matrix M, where M is a n x n matrix whose elements
% are given by the following equation:
% M(r,c) = (r-1)*10*n + 10*c
% Matrix M is initialy defined as a row vector.
M1 = [10:10:10*N];
% Reshape the row vector M in a corresponding n x n square matrix.
M1 = reshape(M1,n,n)'
M2 = M1;
% Get the main diagonal of matrix M.
Diag = diag(M2);
% Get the positions of the main diagonal elements in the original matrix
% M. Keep in mind that the intersect routine requires a row vector version
% of the matrix M, thus the reshape operation is used in order to
% internally trasform matrix M into a row vector.
[Diagonal,DiagonalIndices] = intersect(reshape(M2,1,N),Diag);
% Replace the main diagonal elements with zeros;
M2(DiagonalIndices) = 0;
% Get the upper and lower triangle matrix corresponding to the original
% matrix M.
UpperTriangle = triu(M2);
LowerTriangle = tril(M2);
% Get the non-zero elements positions of the upper and lower triangle matrices.
UpperTriangleNonZeroIndices = find(UpperTriangle~=0);
LowerTriangleNonZeroIndices = find(LowerTriangle~=0);
% Get the non-zero elements of the upper and lower triangle matrices.
NonZeroUpperTriangle = UpperTriangle(UpperTriangleNonZeroIndices)
NonZeroLowerTriangle = LowerTriangle(LowerTriangleNonZeroIndices)
M2(UpperTriangleNonZeroIndices) = NonZeroLowerTriangle;
M2(LowerTriangleNonZeroIndices) = NonZeroUpperTriangle;
M2(DiagonalIndices) = Diagonal;
M2