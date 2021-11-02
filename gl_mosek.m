function [x3, iter3, out3] = gl_mosek(x0, A, b, mu, opts3)
clear prob;
x0;opts3;
[~, res] = mosekopt('symbcon echo(0)');
[m,n]=size(A);[~,l]=size(b);

c=zeros(1,m*l+n*l+n+2);
c(n*l+m*l+1:n*l+m*l+n)=mu;c(m*l+n*l+n+1)=1;
prob.c=c;

A0=zeros(m*l+1,n*l+m*l+n+2);
for t=1:l
    A0(1+(t-1)*m:t*m,1+(t-1)*n:t*n)=A;
    A0(1+(t-1)*m:t*m,1+(t-1)*m+n*l:t*m+n*l)=-eye(m);
end

A0(m*l+1,n*l+m*l+n+2)=1;
prob.a=sparse(A0);
b0=zeros(m*l+1,1);b0(1:m*l)=reshape(b,m*l,1);b0(m*l+1)=1;prob.blc=b0;prob.buc=b0;
cone=repelem([res.symbcon.MSK_CT_QUAD],1+n);cone(n+1)=res.symbcon.MSK_CT_RQUAD;

prob.cones.type=cone;

A0=zeros(n,l+1);A0(:,1)=(m+n)*l+1:(m+n)*l+n;A0(:,2:l+1)=reshape(1:l*n,n,l);
A0=reshape(A0',1,n*(l+1));A0=[A0,n*l+m*l+n+1,n*l+m*l+n+2,n*l+1:(m+n)*l];

prob.cones.sub=A0;

prob.cones.subptr=1:l+1:n*(l+1)+1;

[~, res] = mosekopt('minimize info echo(0)',prob);
x3 = reshape(res.sol.itr.xx(1:n*l), [n,l]); 
iter3 = res.info.MSK_IINF_INTPNT_ITER; 
out3=struct();
out3.fval = res.sol.itr.pobjval;

end