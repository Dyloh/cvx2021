function [x, iter, out] = gl_ADMM_primal(x0, A, b, mu, opts)
if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
if ~isfield(opts, 'sigma'); opts.sigma = 3; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-7; end
if ~isfield(opts, 'gamma'); opts.gamma = 1.618; end
k = 0;
x = x0;
[n,l]=size(x0);
sigma = opts.sigma;
z = zeros(n,l);
y = zeros(n,l);
fp = inf;
nrmC = inf;
f = 0.5*norm(A*x - b,'fro')^2 + mu*sum(norms(x,2,2));
out.fvec = f;
mu1 = mu/sigma;
ATA = A'*A;
ATb = A'*b;
R = chol(ATA + sigma*eye(n));
while k < opts.maxit && abs(f - fp) > opts.ftol && nrmC > opts.gtol
    fp = f;    
    w = ATb + sigma*(y+z);
    x = R\(R'\w);
    y = prox(x-z, mu1);
    c = x-y;
    z = z - opts.gamma * c;
    f = 0.5*norm(A*x - b,'fro')^2 + mu*sum(norms(x,2,2));
    nrmC = norm(c,'fro');
    k = k + 1;
    out.fvec = [out.fvec; f];
end
out.fval = f;
iter = k;
end
function y = prox(x, mu)
nmx = norms(x,2,2);
y = (1 - mu./max(nmx,mu)).*x;
end