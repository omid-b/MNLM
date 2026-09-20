function [DATA,xx,yy] = transform_in(Data,y,x) 
data=Data(:,3);X=Data(:,1);Y=Data(:,2);
M=x;N=y;
for k=1:N
DATA(k,:)=data(1+((k-1)*M):k*M);
xx(k,:)=X(1+((k-1)*M):k*M);
yy(k,:)=Y(1+((k-1)*M):k*M);
end