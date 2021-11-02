function [x, iter, out] = gl_FGD_primal(x0, A, b, mu0, opts)

if ~isfield(opts, 'maxit'); opts.maxit = 30; end
if ~isfield(opts, 'maxit_inn'); opts.maxit_inn = 1000; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'factor'); opts.factor = 0.1; end
if ~isfield(opts, 'verbose'); opts.verbose = 0; end
if ~isfield(opts, 'mu1'); opts.mu1 = 100; end
if ~isfield(opts, 'gtol_init_ratio'); opts.gtol_init_ratio = 1/opts.gtol; end
if ~isfield(opts, 'ftol_init_ratio'); opts.ftol_init_ratio = 1e5; end
if ~isfield(opts, 'opts1'); opts.opts1 = struct(); end
if ~isfield(opts, 'etaf'); opts.etaf = 1e-1; end
if ~isfield(opts, 'etag'); opts.etag = 1e-1; end
L = eigs(A'*A,1);
if ~isfield(opts, 'alpha0'); opts.alpha0 = 1/L; end


out = struct();
out.fvec = [];
k = 0;
x = x0;
mu_t = opts.mu1;
f = Func(A, b, mu_t, x);
opts1 = opts.opts1;
opts1.ftol = opts.ftol*opts.ftol_init_ratio;
opts1.gtol = opts.gtol*opts.gtol_init_ratio;
out.itr_inn = 0;

while k < opts.maxit
    opts1.maxit = opts.maxit_inn;
    opts1.gtol = max(opts1.gtol * opts.etag, opts.gtol);
    opts1.ftol = max(opts1.ftol * opts.etaf, opts.ftol);
    opts1.verbose = opts.verbose > 1;
    opts1.alpha0 = opts.alpha0;
    opts1.thres = 10^(-3)*mu_t;
    fp = f;
    [x, out1] = gl_FGD_inner(x, A, b, mu_t, mu0, opts1);
    f = out1.fvec(end);
    out.fvec = [out.fvec, out1.fvec];
    out.itr_inn = out.itr_inn + out1.itr;
    k = k + 1;
    nrmG = out1.nrmG;
    if opts.verbose
        fprintf('itr: %d\tmu_t: %e\titr_inn: %d\tfval: %e\tnrmG: %.1e\n', k, mu_t, out1.itr, f, nrmG);
    end
    if ~out1.flag
        mu_t = max(mu_t * opts.factor, mu0);
    end
    if mu_t == mu0 && (nrmG < opts.gtol || abs(f-fp) < opts.ftol)
        break;
    end
    
end
iter = out.itr_inn;
x(abs(x)<4*1e-6)=0;
out.fval = Func(A,b,mu0,x);
end



function f = Func(A, b, mu0, x)
    w = A * x - b;
    f = 0.5 * norm(w,'fro')^2 + mu0 * sum(norms(x, 2, 2));
end


function [x, out] = gl_FGD_inner(x, A, b, mu, mu0, opts)

if ~isfield(opts, 'maxit'); opts.maxit = 200; end
if ~isfield(opts, 'sigma'); opts.sigma = mu/10000; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'verbose'); opts.verbose = 0; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = norm(A,2)^-2; end
if ~isfield(opts, 'ls'); opts.ls = 0; end
if ~isfield(opts, 'bb'); opts.bb = 1; end

r = A * x - b;
g = A' * r;
xnorm = norms(x,2,2);

idx = xnorm < opts.sigma;
xnorm(idx)=opts.sigma;
huber_g = x./xnorm;
g = g + mu * huber_g;

xnorm = norms(x,2,2);

nrmG = norm(g,'fro');
f = .5*norm(r,'fro')^2 + mu*(sum(xnorm(idx).^2/(2*opts.sigma)) + sum(xnorm(xnorm >= opts.sigma)) - opts.sigma/2);

out = struct();
out.fvec = .5*norm(r,'fro')^2 + mu0*sum(xnorm);
out.fvec=[];
alpha = opts.alpha0;
eta = 0.2;
rhols = 1e-6;
gamma = 0.85;
Q = 1;
Cval = f;
y = x;
k = 0;
fp = inf;
xp = x;
while k < opts.maxit && nrmG > opts.gtol && abs(f - fp) > opts.ftol
    fp = f;
    gp = g;
    yp = y;
    theta = (k-1)/(k+2);
    y = x + theta * (x - xp);
    xp = x;
    r = A * y - b;
    g = A' * r;
    xnorm = norms(y,2,2);
    
    idx = xnorm < opts.sigma;
    xnorm(idx)=opts.sigma;
    huber_g = y./xnorm;
    g = g + mu * huber_g;
    
    if opts.bb
        dy = y - yp;
        dg = g - gp;
        dyg = abs(sum(dot(dy,dg)));
        if dyg > 0
            if mod(k,2) == 0
                alpha = norm(dy,'fro')^2/dyg;
            else
                alpha = dyg/norm(dg,'fro')^2;
            end
        end
        alpha = min(max(alpha,1e-12),1e12);
    else
        alpha = opts.alpha0;
    end
    x = y - alpha * g;
    if opts.bb && opts.ls
        nls = 1;
        while 1
        
            xnorm = norms(x,2,2);
            f = .5*norm(r,'fro')^2 + mu*(sum(xnorm(idx).^2/(2*opts.sigma)) + sum(xnorm(xnorm >= opts.sigma)) - opts.sigma/2);
            if f <= Cval - alpha*rhols*nrmG^2 || nls >= 5
                break
            end
            alpha = eta*alpha;
            nls = nls+1;
            x = y - alpha * g;
        end
    else
        f = 0.5 * norm(A*x - b, 2)^2 + mu0*sum(norms(x,2,2));
    end
    nrmG = norm(g,'fro');
    k = k + 1;
    out.fvec = [out.fvec, f];
    if opts.verbose
        fprintf('itr: %d\tt: %e\tfval: %e\tnrmG: %e\n', k, t, f, nrmG);
    end

    if k >8 && min(out.fvec(k-7:k)) > out.fvec(k-8)
        break;
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
