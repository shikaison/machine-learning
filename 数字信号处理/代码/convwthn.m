function [y,ny] = convwthn(x,nx,h,nh)
y = conv(x,h);
ny1 = nx(1) + nh(1);
ny2 = nx(end) + nh(end);
ny = ny1:ny2;
end