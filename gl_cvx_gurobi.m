function [x, iter1, out1] =gl_cvx_gurobi(x0, A, b, mu, opts1)
[n,l]=size(x0);
cvx_quiet 'True'
cvx_begin
    
    cvx_solver gurobi
    variable x(n,l)
    minimize (0.5*square_pos(norm(A*x-b,'fro'))+mu*sum(norms(x,2,2)));
cvx_end
%y=cvx_optival;
out1=struct();
out1.fval=cvx_optval;
iter1=0;opts1;
end
