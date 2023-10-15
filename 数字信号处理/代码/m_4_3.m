clc;clear all;
close all;
x = [1,2,3,4,5];
nx = 0:4;
h = [-5,4,7,9];
nh = -2:1;
figure;
subplot(2,1,1);
stem(nx,x);
grid on;
subplot(2,1,2);
stem(nh,h);

figure;
[y1,ny1] = seqadd(x,nx,h,nh);
stem(ny1,y1);

figure;
[y2,ny2] = seqmult(x,nx,h,nh);
stem(ny2,y2);

figure;
[y3,ny3] = convwthn(x,nx,h,nh);
stem(ny3,y3);

