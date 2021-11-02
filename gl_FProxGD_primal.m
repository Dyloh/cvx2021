function [x, iter, out] = gl_FProxGD_primal(x0, A, b, mu, opts)

if ~isfield(opts, 'maxit'); opts.maxit = 30; end
if ~isfield(opts, 'maxit_inn'); opts.maxit_inn = 200; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-10; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-8; end
if ~isfield(opts, 'factor'); opts.factor = 0.1; end
if ~isfield(opts, 'mu1'); opts.mu1 = 100; end
if ~isfield(opts, 'opts1'); opts.opts1 = struct(); end
if ~isfield(opts, 'thres'); opts.thres = 1e-7; end
if ~isfield(opts, 'bb'); opts.bb = 1; end
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
    opts1.sigma = mu_t * 1e-3;
    fp = f;
    opts1.bb=opts.bb;
    [x, out1] = fgrad_inn(x, A, b, mu_t, mu, opts1);
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
x(abs(x)<opts.thres) = 0;
out.fval = 0.5 * norm(A * x - b,'fro')^2 + mu * sum(vecnorm(x,2,2));
end

function [x, out] = fgrad_inn(x0, A, b, mu_t, mu, opts)

if ~isfield(opts, 'maxit'); opts.maxit = 200; end
if ~isfield(opts, 'ftol'); opts.ftol = 1e-8; end
if ~isfield(opts, 'gtol'); opts.gtol = 1e-6; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = 1e-3; end
if ~isfield(opts, 'ls'); opts.ls = 1; end
if ~isfield(opts, 'bb'); opts.bb = 1; end
if ~isfield(opts, 'sigma'); opts.sigma = 0.1; end


k = 0;
t = opts.alpha0;
x = x0;
y = x;
xp = x0;

fp = inf;
r = A*x0 - b;
g = A'*r;
tmp = 0.5*norm(r,'fro')^2;
f =  tmp + mu*sum(vecnorm(x,2,2));

nrmG = norm(x - prox(x - g, mu_t),'fro');
out = struct();
out.fvec = [];%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5



while k < opts.maxit && nrmG > opts.gtol && abs(f - fp) > opts.ftol
    

    
    gp = g;
    yp = y;
    fp = f;

    
    theta = (k - 1) / (k + 2);
    y = x + theta * (x - xp);
    xp = x;
    r = A * y - b;
    g = A' * r;
   
    
    if opts.bb && opts.ls
        dy = y - yp;
        dg = g - gp;
        dyg = sum(dy.*dg,'all');
        
        if dyg > 0
            if mod(k,2) == 0
                t = norm(dy,'fro')^2/dyg;
            else
                t = dyg/norm(dg,'fro')^2;
            end
        end

        t = min(max(t,opts.alpha0),1e12);
        
       
        
    else
        t = opts.alpha0;
    end    
    x = prox(y - t * g, t * mu_t);
    if opts.ls
        nls = 1;
        while 1
            tmp = 0.5 * norm(A*x - b, 'fro')^2;
            if tmp <= 0.5*norm(r,'fro')^2 + sum(g.*(x-y),'all') + 0.5/t*norm(x-y,'fro')^2 || nls == 5
                
                break;
            end
            
            t = 0.3*t; nls = nls + 1;
            x = prox(y - t * g, t * mu_t);
        end
        
        
        
        f = tmp + mu*sum(vecnorm(x,2,2));
        
       
        
    else
        f = 0.5 * norm(A*x - b, 2)^2 + mu*sum(vecnorm(x,2,2));
    end
    
    
    
    nrmG = norm(x - y,'fro')/t;
    
    k = k + 1;
    out.fvec = [out.fvec, f];
    if k > 8 && min(out.fvec(k-7:k)) > out.fvec(k-8)
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