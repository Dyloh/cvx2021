function [x, iter,out] = gl_ProxGD_primal(x0, A, b, mu0, opts)

if ~isfield(opts, 'maxit'); opts.maxit = 30; end
if ~isfield(opts, 'maxit_inn'); opts.maxit_inn = 200; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-12; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-7; end
if ~isfield(opts, 'factor'); opts.factor = 0.1; end
if ~isfield(opts, 'verbose'); opts.verbose = 0; end
if ~isfield(opts, 'mu1'); opts.mu1 = 100; end
if ~isfield(opts, 'gtol_init_ratio'); opts.gtol_init_ratio = 1/opts.gtol; end
if ~isfield(opts, 'ftol_init_ratio'); opts.ftol_init_ratio = 1e5; end
if ~isfield(opts, 'opts1'); opts.opts1 = struct(); end
if ~isfield(opts, 'etaf'); opts.etaf = 1e-1; end
if ~isfield(opts, 'etag'); opts.etag = 1e-1; end
if ~isfield(opts, 'method'); opts.method = 'grad_huber'; end
L = eigs(A'*A,1);
if ~isfield(opts, 'alpha0'); opts.alpha0 = 1/L; end
if ~isfield(opts, 'bb'); opts.bb = 1; end


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
    opts1.bb=opts.bb;
    if strcmp(opts.method, 'grad_huber'); opts1.thres = 1e-3*mu_t; end
    fp = f;
    [x, out1] = LASSO_proximal_grad_inn(x, A, b, mu_t, mu0, opts1);
    f = out1.fvec(end);
    out.fvec = [out.fvec, out1.fvec];
    k = k + 1;
    out.itr_inn = out.itr_inn + out1.itr;
    nrmG = norm(x - prox(x - A'*(A*x - b),mu0),'Fro');
    

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



out.fval = f;
out.itr = k;
iter=out.itr_inn;


   

end


 function f = Func(A, b, mu0, x)
    w = A * x - b;
    f = 0.5 * norm(w,'fro')^2 + mu0 * sum(norms(x,2,2));
end



function [x, out] = LASSO_proximal_grad_inn(x0, A, b, mu, mu0, opts)

if ~isfield(opts, 'maxit'); opts.maxit = 10000; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-12; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 1e-3; end
if ~isfield(opts, 'ls'); opts.ls = 1; end
if ~isfield(opts, 'bb'); opts.bb = 1; end


out = struct();
k = 0;
x = x0;
t = opts.alpha0;

fp = inf;
r = A*x0 - b;
g = A'*r;
tmp = .5*norm(r,'fro')^2;
f =  tmp + mu0*sum(norms(x,2,2));
nrmG = norm(x - prox(x - g,mu),2);
out.fvec = tmp + mu0*sum(norms(x,2,2));

Cval = tmp; Q = 1; gamma = 0.85;

while k < opts.maxit && nrmG > opts.gtol && abs(f - fp) > opts.ftol
    
    gp = g;
    fp = f;
    xp = x;
    

    x = prox(xp - t * g, t * mu);
   
    if opts.ls
        nls = 0;
        while 1
            tmp = 0.5 * norm(A*x - b, 'fro')^2;
            if tmp <= Cval + sum(sum(g.*(x-xp))) + .5/t*norm(x-xp,'fro')^2 || nls == 5
                break;
            end
            
            t = 0.2*t; nls = nls + 1;
            x = prox(xp - t * g, t * mu);
        end
        

        f = tmp + mu0*sum(norms(x,2,2));
        

    else
        f = 0.5 * norm(A*x - b, 'fro')^2 + mu0*sum(norms(x,2,2));
    end

    nrmG = norm(x - xp,'fro')/t;
    r = A * x - b;
    g = A' * r;
    

    if opts.bb && opts.ls
        dx = x - xp;
        dg = g - gp;
        dxg = abs(sum(sum(dx.*dg)));
        if dxg > 0
            if mod(k,2) == 0
                t = norm(dx,'fro')^2/dxg;
                %t = sum(dx.*dx)/dxg;
            else
                %t = norm(dx,'fro')^2/dxg;
                t = dxg/norm(dg,'fro')^2;
            end
        end
        

        t = min(max(t,opts.alpha0),1e12);
        Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + tmp)/Q;
        

    else
        t = opts.alpha0;
    end
    

    k = k + 1;
    out.fvec = [out.fvec, f];
    if opts.verbose
        fprintf('itr: %d\tt: %e\tfval: %e\tnrmG: %e\n', k, t, f, nrmG);
    end
    

    if k > 8 && min(out.fvec(k-7:k)) - out.fvec(k-8) > opts.ftol
        break;
    end
end


if k == opts.maxit
    out.flag = 1;
else
    out.flag = 0;
end
out.fvec = out.fvec(1:k);
out.fval = f;
out.itr = k;
out.nrmG = nrmG;
end

function y = prox(x, mu)
nmx = vecnorm(x,2,2);
y = (1 - mu./max(nmx,mu)).*x;
end

