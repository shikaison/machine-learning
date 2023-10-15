function [] = cycle(x,k)
N = length(x);
nx = 0:N-1;
ny = 0:(k*N-1);
y = x(mod(ny,N)+1);
figure;
subplot(2,1,1);
stem(nx,x,'.');
axis([-1,N+1,0,5]);
grid;
subplot(2,1,2);
stem(ny,y,'.');
axis([-1,k*N,0,5]);
grid;
end