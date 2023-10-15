function [x,n] = impseq(n0,n1,n2)
if ((n0<n1)|(n0>n2)|(n1>n2))
    error('参数必须满足n1<=n0<=n2')
end
n = [n1:n2];
x = [(n-n0)==0];
stem(n,x);
title('单位抽样序列');
end