function [x, iter, out] = gl_ALM_dual(x0, A, b, mu, opts)
if ~isfield(opts, 'maxit'); opts.maxit = 200; end
if ~isfield(opts, 'maxit_inn'); opts.maxit_inn = 3; end
if ~isfield(opts, 'sigma'); opts.sigma = 5; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-7; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
k = 0;
x = x0;
out.itr_inn = 0;
f = 0.5*norm(A*x - b,'fro')^2 + mu*sum(norms(x,2,2));
fp = inf; 
nrmC = inf;
f0 = f;
out.fvec = f0;
z = zeros(size(b));
sigma = opts.sigma;
opts1.maxit = opts.maxit_inn;
opts1.ftol = opts.ftol;
opts1.gtol = opts.gtol;
while k < opts.maxit  && abs(f - fp) > opts.ftol && nrmC > opts.gtol
    fp = f;
    xsigma = x/sigma;
    [z, out1] = grad_inn(z,xsigma,A,b,sigma,mu,opts1);
    ATz = A'*z;
    w = proj( ATz-xsigma, mu);
    c = ATz-w;
    x = x - sigma*c;
    nrmC = norm(c,'fro');
    f = 0.5*norm(A*x - b,'fro')^2 + mu*sum(norms(x,2,2));
    out.fvec = [out.fvec;f];
    k = k + 1;
    out.itr_inn = out.itr_inn + out1.itr;
end

out.fval = f;
iter = k;
end

function [z, out] = grad_inn(z0,x,A,b,sigma,mu,opts)
if ~isfield(opts, 'maxit'); opts.maxit = 200; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 0.01; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-5; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-7; end
z = z0;
y = A'*z - x;
nrmy = norms(y,2,2);
g = z + b + sigma*A*((1-mu./max(nrmy,mu)).*y);
f = 0.5*norm(z,'fro')^2 + sum(b.*z,'all') + sigma/2*sum(max(nrmy-mu,0).^2);
out = struct();
alpha = opts.alpha0;
eta = 0.2;
rhols = 1e-6;
gamma = 0.85;
Q = 1;
Cval = f;
for k = 1:opts.maxit 
    fp = f;
    gp = g;
    zp = z;
    nls = 1;
    while 1
        z = zp-alpha*gp;
        y = A'*z-x;
        nrmy = norms(y,2,2);
        g = z+b+sigma*A*((1-mu./max(nrmy,mu)).*y);
        f = 0.5*norm(z,'fro')^2 + sum(b.*z,'all') + sigma/2*sum(max(nrmy-mu,0).^2);
        nrmG = norm(g,'fro');
        if f <= Cval - alpha*rhols*nrmG^2 || nls >= 10
            break
        end
        alpha = eta*alpha;
        nls = nls+1;
    end
    nrmG = norm(g,'fro');
    if nrmG < opts.gtol || abs(fp - f) < opts.ftol
        break;
    end
    dz = z - zp;
    dg = g - gp;
    dxg = sum(dz.*dg,'all');
    if dxg > 0
        if mod(k,2)==0
            alpha = norm(dz,'fro')^2/dxg;
        else
            alpha = dxg/norm(dg,'fro')^2;
        end
        alpha = max(min(alpha, 1e12), 1e-12);
    end
    Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + f)/Q;
end
out.itr = k;

end
function w = proj(x, mu)
nmx = norms(x,2,2);
w = min(1,mu./nmx).*x;
end