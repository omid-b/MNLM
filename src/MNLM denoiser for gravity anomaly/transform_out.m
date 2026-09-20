function [CC] = transform_out(data,y,x) 
x=x;y=y;
M=length(x);N=length(y);
for k=1:N
BB(1+((k-1)*M):k*M)=data(k,:)';
XX(1+((k-1)*M):k*M)=x;
YY(1+((k-1)*M):k*M)=y(k);
end
BB=BB';
XX=XX';
YY=YY';
CC=[XX YY BB];
