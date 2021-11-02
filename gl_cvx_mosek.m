function [x, iter1, out1] =gl_cvx_mosek(x0, A, b, mu, opts1)
[n,l]=size(x0);
cvx_quiet 'True'
cvx_begin
    
    cvx_solver mosek
    variable x(n,l)
    minimize (0.5*square_pos(norm(A*x-b,'fro'))+mu*sum(norms(x,2,2)));
cvx_end
%y=cvx_optival;
out1=struct();
iter1=0;out1.fval=cvx_optval;opts1;
end
