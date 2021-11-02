function [x, iter, out] = gl_ADMM_dual(x0, A, b, mu, opts)
if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
if ~isfield(opts, 'sigma'); opts.sigma = 0.55; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-7; end
if ~isfield(opts, 'gamma'); opts.gamma = 1.618; end
k = 0;
x = x0;
[m,l] = size(b);
sigma = opts.sigma;
z = zeros(m,l);
f = 0.5*norm(A*x - b,'fro')^2 + mu*sum(norms(x,2,2));
fp = inf;
nrmC = inf;
out.fvec = f;
R = chol(eye(m)+sigma*(A*A'));
ATz = A'*z;
while k < opts.maxit && abs(f - fp) > opts.ftol && nrmC > opts.gtol
    fp = f;
    w = proj(ATz-x/sigma,mu);
    h = A * (w*sigma+x)-b;
    z = R\(R'\h);
    ATz = A'*z;
    c = ATz - w;
    x = x - opts.gamma * sigma * c;
    nrmC = norm(c,'fro');
    f = 0.5*norm(A*x-b,'fro')^2 + mu*sum(norms(x,2,2));
    k = k + 1;
    out.fvec = [out.fvec; f];
end
out.fval = f;
iter = k;
end
function w = proj(x, mu)
nmx = norms(x,2,2);
w = min(1,mu./nmx).*x;
end