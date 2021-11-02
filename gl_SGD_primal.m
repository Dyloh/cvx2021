function [x4, iter4, out4]=gl_SGD_primal(x0, A, b, mu0, opts)
mu=100;
eta=0.1;
iter4=0;
out4 = struct();
out4.fvec = [];
t=0;
while mu > mu0
    opts.times=t;
    [x0, out]=LASSO_subgrad_inn(x0, A, b, mu, mu0, opts);
    mu=mu*eta;
    iter4=iter4+out.itr;
    out4.fvec = [out4.fvec,out.fvec];
    t=t+1;
end
    mu=mu0;
    opts.times=t+2;
    [x4, out]=LASSO_subgrad_inn(x0, A, b, mu, mu0, opts);
    iter4=iter4+out.itr;
    out4.fvec = [out4.fvec,out.fvec];
    %x4(abs(x4)<0.0001)=0;
    r = A * x4 - b;
    out4.fval= .5*norm(r,'fro')^2 + mu0*sum(norms(x4,2,2));
end

function [x0, out] = LASSO_subgrad_inn(x0, A, b, mu, mu0, opts)
if ~isfield(opts, 'maxit'); opts.maxit = 3000; end
if ~isfield(opts, 'thres'); opts.thres = 1e-4; end
if ~isfield(opts, 'step_type'); opts.step_type = 'fixed'; end
if ~isfield(opts, 'alpha0'); opts.alpha0 = norm(A,2)^-2; end
if ~isfield(opts, 'ftol'); opts.ftol = 10^(-4-opts.times); end

if mu > mu0
    opts.step_type = 'fixed';
else
    opts.step_type = 'diminishing';
end

out = struct();
out.fvec = [];
r = A * x0 - b;
gx = A' * r;
sub0 = norms(x0,2,2);
sub0 = sub0 + (sub0==0);
sub_g = gx + mu * x0./sub0;
f_best = inf;

for k = 1:opts.maxit
    alpha = set_step(k, opts);
    x0 = x0 - alpha * sub_g;
    r = A * x0 - b;
    g = A' * r;

    x0(abs(x0) < opts.thres) = 0;
    sub0 = norms(x0,2,2);
    sub0 = sub0 + (sub0==0);
    sub_g = g + mu * x0./sub0;

    out.grad_hist(k) = norm(r, 'Fro');
    tmp = .5*norm(r,'Fro')^2;
    nrmx1 = sum(norms(x0,2,2));
    f = tmp + mu * nrmx1;
   
    out.f_hist(k) = f;

    f_best = min(f_best, f);
    out.f_hist_best(k) = f_best;
    out.fvec = [out.fvec, tmp + mu0*nrmx1];

    FDiff = abs(out.f_hist(k) - out.f_hist(max(k-1,1))) / abs(out.f_hist_best(1));
    BFDiff = abs(out.f_hist_best(max(k - 8,1)) - min(out.f_hist_best(max(k-7,1):k)));
    if (k > 1 && FDiff < opts.ftol) || (k > 8 && BFDiff < opts.ftol)
        break;
    end
end


if k == opts.maxit
    out.flag = 1;
else
    out.flag = 0;
end

out.itr = k;
end

function a = set_step(k, opts)
type = opts.step_type;
if strcmp(type, 'fixed')
    a = opts.alpha0;
elseif strcmp(type, 'diminishing')
    a = opts.alpha0 / sqrt(max(k,100)-99);
elseif strcmp(type, 'diminishing2')
    a = opts.alpha0 / (max(k,100)-99);
else
    error('unsupported type.');
end
end
