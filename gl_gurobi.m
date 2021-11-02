function [x4, iter4, out4] = gl_gurobi(x0, A, b, mu, opts4)
clear prob;
params.outputflag = 0;
[m,n]=size(A);[~,l]=size(b);
s=m*l+n*l+n+1;
c=zeros(s,1);
c(n*l+m*l+1:n*l+m*l+n)=mu;c(m*l+n*l+n+1)=1/2;
model.obj=c;
model.sense = '=';
A0=zeros(m*l,s);
for t=1:l
    A0(1+(t-1)*m:t*m,1+(t-1)*n:t*n)=A;
    A0(1+(t-1)*m:t*m,1+(t-1)*m+n*l:t*m+n*l)=-eye(m);
end
model.A=sparse(A0);
model.rhs=reshape(b,m*l,1);
model.lb = [-inf(n*l+m*l,1); zeros(1+n,1)];


for i = 1 : n 
    
    d = zeros(s, 1);
    d(i:n:n*(l-1)+i) = 1;  d((n+m)*l+i) = -1;
    model.quadcon(i).Qc=spdiags(d,0,s,s);
    model.quadcon(i).q=zeros(s, 1);
    model.quadcon(i).rhs=0;
end
i=n+1;
d = zeros(s, 1);
d(n*l+1:(n+m)*l) = 1; 
model.quadcon(i).Qc = spdiags(d,0,s,s);
d = zeros(s, 1);d(s)=-1;
model.quadcon(i).q = d;
model.quadcon(i).rhs = 0;

params.outputflag = 0;
result = gurobi(model,params);
x4 = reshape(result.x(1:n*l), [n,l]);
iter4 = result.baritercount;
out4=struct();
out4.fval = result.objval;




end