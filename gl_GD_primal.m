function [x, iter, out] = gl_GD_primal(x0, A, b, mu, opts)

if ~isfield(opts, 'maxit'); opts.maxit = 30; end
if ~isfield(opts, 'maxit_inn'); opts.maxit_inn = 200; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-10; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-8; end
if ~isfield(opts, 'factor'); opts.factor = 0.1; end
if ~isfield(opts, 'mu1'); opts.mu1 = 100; end
if ~isfield(opts, 'opts1'); opts.opts1 = struct(); end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 0.01; end
if ~isfield(opts, 'thres'); opts.thres = 1e-10; end

k = 0; iter = 0;
x = x0;
mu_t = opts.mu1;
f = 0.5 * norm(A * x - b,'fro')^2 + mu * sum(vecnorm(x,2,2));
out = struct();
out.fvec = [];
opts1 = opts.opts1;
while k < opts.maxit   

    opts1.maxit = opts.maxit_inn;
    opts1.gtol = max(10^-(1+k), opts.gtol);
    opts1.ftol = max(10^-(4+k), opts.ftol);
    opts1.alpha0 = opts.alpha0;
    opts1.sigma = mu_t * 10^(-3-0.2*k);
    
    %%%
    fp = f;
    [x, out1] = grad_inn(x, A, b, mu_t, mu, opts1);
    f = out1.fvec(end);
    out.fvec = [out.fvec, out1.fvec];
    k = k + 1;
    nrmG = out1.nrmG;
    iter = iter + out1.itr;
    

    if ~out1.flag
        mu_t = max(mu_t * opts.factor, mu);
    end

    if mu_t == mu && (nrmG < opts.gtol || abs(f-fp) < opts.ftol)
        break;
    end
end
%x(abs(x)<opts.thres) = 0;
out.fval = 0.5 * norm(A * x - b,'fro')^2 + mu * sum(vecnorm(x,2,2));
end

function [x, out] = grad_inn(x, A, b, mu_t, mu, opts)

if ~isfield(opts, 'maxit'); opts.maxit = 200; end
if ~isfield(opts, 'sigma'); opts.sigma = 0.1; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 0.01; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end


r = A * x - b;
g = A' * r;
nmx = vecnorm(x,2,2);
huber_g = x ./ max(nmx,opts.sigma);
g = g + mu_t * huber_g;
nrmG = norm(g,2);
idx1 = nmx <= opts.sigma;
idx2 = nmx > opts.sigma;
f = 0.5*norm(r,'fro')^2 + mu_t*(norm(x(idx1,:),'fro')^2/(2*opts.sigma) + ...
    sum(vecnorm(x(idx2,:),2,2)) - opts.sigma/2 * length(idx2));

out = struct();
out.fvec = [];

alpha = opts.alpha0;
eta = 0.2;


rhols = 1e-6;
gamma = 0.85;
Q = 1;
Cval = f;


for k = 1:opts.maxit

    fp = f;
    gp = g;
    xp = x;

    nls = 1;
    while 1   
        x = xp - alpha*gp;
        r = A * x - b;
        g = A' * r;
        nmx = vecnorm(x,2,2);
        huber_g = x ./ max(nmx,opts.sigma);
        g = g + mu_t * huber_g;
        idx1 = nmx <= opts.sigma;
        idx2 = nmx > opts.sigma;
        f = 0.5*norm(r,'fro')^2 + mu_t*(norm(x(idx1,:),'fro')^2/(2*opts.sigma) + ...
            sum(vecnorm(x(idx2,:),2,2)) - opts.sigma/2 * length(idx2));
        

        if f <= Cval - alpha*rhols*nrmG^2 || nls >= 10
            break
        end
        alpha = eta*alpha;
        nls = nls+1;
    end

    
    nrmG = norm(g,'fro');
    forg = 0.5*norm(r,'fro')^2 + mu*sum(nmx);
    out.fvec = [out.fvec, forg];

    if nrmG < opts.gtol || abs(fp - f) < opts.ftol
        break;
    end
    
    dx = x - xp;
    dg = g - gp;
    dxg = sum(dx.*dg,'all');
    if dxg > 0
        if mod(k,2)==0
            alpha = norm(dx,'fro')^2/dxg;
        else
            alpha = dxg/norm(dg,'fro')^2;
        end

        alpha = max(min(alpha, 1e12), 1e-12);
    end

    Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + f)/Q;
end

if k == opts.maxit
    out.flag = 1;
else
    out.flag = 0;
end

out.fval = f;
out.itr = k;
out.nrmG = nrmG;
end