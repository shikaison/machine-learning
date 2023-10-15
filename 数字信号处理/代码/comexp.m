function [x,n] = comexp(a,b,n0,n1)
n = n0:n1;
x = exp((a+b*1i)*n);
figure;
subplot(2,1,1);
stem(n,real(x),'.');
axis([-4,10,min(real(x))-1,1.2*max(real(x))]);
title('复指数序列');
ylabel('实部');
grid;
subplot(2,1,2);
stem(n,imag(x),'.');
axis([-4,10,min(imag(x))-1,1.2*max(imag(x))]);
ylabel('虚部');
xlabel('n');
grid;
end