%calculate sqrtm inv of a matrix
function out=mysqrtinv(W);
%W= p x p matrix

[U,D]=svd(W);
d=diag(D).^(-1/2);
out=U*diag(d)*U';

