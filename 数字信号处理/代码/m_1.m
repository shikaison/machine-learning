clc;clear all;close all;
n1 = -5;
n0 = 0;
n2 = 5;
figure;
[y1,ny1] = impseq(n0,n1,n2);
stem(ny1,y1);
axis([-6,6,0,2]);

figure;
[y2,ny2] = stepseq(n0,n1,n2);
stem(ny2,y2);
axis([-6,6,0,2]);

figure;
[y3,ny3] = RN(2,n1,n2);
stem(ny3,y3);
axis([-6,6,0,2]);

figure;
[y4,ny4] = realexp(0.8,0,10);
stem(ny4,y4);
axis([-1,11,0,10]);

[y5,ny5] = comexp(0.4,0.6,-1,10);
