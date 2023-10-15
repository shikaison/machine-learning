clc;clear all;close all;
x = [1,2,3,4,5,6,7,8,9];
n = 0:8;
figure;
stem(n,x);

cycle(x,3);

[y,ny] = seqshift(x,n,3);
figure;
stem(ny,y);

[y1,ny1] = seqshift(x,n,-2);
figure;
stem(ny1,y1);

[y2,ny2] = seqfold(x,n);
figure;
stem(ny2,y2);