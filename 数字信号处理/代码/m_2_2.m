clc;clear;
x1 = [1,3,5,7,6,4,2,1];
ns1 = -3;
x2 = [4,0,2,1,-1,3];
ns2 = 1;
nf1 = ns1 + length(x1) - 1;
nf2 = ns2 + length(x2) - 1;
n1 = ns1:nf1;
n2 = ns2:nf2;
n = min(ns1,ns2):max(nf1,nf2);
y1 = zeros(1,length(n));
y2 = y1;
y1(find((n >= ns1)&(n<=nf1) == 1)) = x1;
y2(find((n >= ns2)&(n<=nf2) == 1)) = x2;
ya = y1 + y2;
ym = y1 .* y2;

subplot(2,2,1);
stem(n1,x1,'.');
ylabel('x1(n)');
grid;

subplot(2,2,2);
stem(n2,x2,'.');
xlabel('n');
ylabel('x2(n)');
grid;

subplot(2,2,3);
stem(n,ya,'.');
ylabel('y1(n)+ya(n)');
grid;

subplot(2,2,4);
stem(n1,x1,'.');
xlabel('n');
ylabel('y1(n)*y2(n)');
grid;
