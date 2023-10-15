function [x,n] = RN(np1,ns,nf)
N = np1;
n = ns:nf;
np = 0;
x = stepseq(np,ns,nf) - stepseq(np1,ns,nf);
stem(n,x);
title('矩形序列RN');
end