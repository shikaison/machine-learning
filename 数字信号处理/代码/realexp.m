function [x,n] = realexp(a,n0,n1)
n = n0:n1;
x = a.^n;
stem(n,x);
title('实指数序列');
end
