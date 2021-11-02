% function Test_group_lasso

% min 0.5 * ||A * x - b||_2^2 + mu * ||x||_{1,2}

% generate data
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
n = 512;
m = 256;
A = randn(m,n);
k = round(n*0.1); l = 2;
A = randn(m,n);
p = randperm(n); p = p(1:k);
u = zeros(n,l);  u(p,:) = randn(k,l);  
b = A*u;
mu = 1e-2;

fig = figure(1);
subplot(2,1,1); 
% plot(1:n, u(:,1), '*');
plot(1:n, u(:,1), '*', 1:n, u(:,2), 'o'); 
xlim([1 n])
title('(1) exact solution $u$', 'Interpreter','latex');

x0 = randn(n, l);

errfun = @(x1, x2) norm(x1 - x2, 'fro') / (1 + norm(x1,'fro'));
errfun_exact = @(x) norm(x - u, 'fro') / (1 + norm(u,'fro'));
sparisity = @(x) sum(abs(x(:)) > 1E-6 * max(abs(x(:)))) /(n*l);

% cvx calling mosek
opts1 = []; % modify options
tic;
[x1, iter1, out1] = gl_cvx_mosek(x0, A, b, mu, opts1);
t1 = toc;

% cvx calling gurobi
opts2 = []; % modify options
tic;
[x2, iter2, out2] = gl_cvx_gurobi(x0, A, b, mu, opts2);
t2 = toc;

% call mosek directly
opts3 = []; % modify options
tic;
[x3, iter3, out3] = gl_mosek(x0, A, b, mu, opts3);
t3 = toc;

% call gurobi directly
opts4 = []; % modify options
tic;
[x4, iter4, out4] = gl_gurobi(x0, A, b, mu, opts4);
t4 = toc;

% other approaches
% % Projected Gradient Descent Method
% opts5 = []; % modify options
% tic;
% [x5, iter5, out5] = gl_PGD_primal(x0, A, b, mu, opts5);
% t5 = toc;
% 
% % Projected Gradient Descent Method
% opts6 = []; % modify options
% tic;
% [x6, iter6, out6] = gl_SGD_primal(x0, A, b, mu, opts6);
% t6 = toc;

% Gradient Method with Smoothing
opts7 = []; % modify options
tic;
[x7, iter7, out7] = gl_GD_primal(x0, A, b, mu, opts7);
t7 = toc;

% Fast Gradient Method with Smoothing
opts8 = []; % modify options
tic;
[x8, iter8, out8] = gl_FGD_primal(x0, A, b, mu, opts8);
t8 = toc;

% Proximal Gradient Method
opts9 = []; % modify options
tic;
[x9, iter9, out9] = gl_ProxGD_primal(x0, A, b, mu, opts9);
t9 = toc;

% Fast Proximal Gradient Method
opts10 = []; % modify options
tic;
[x10, iter10, out10] = gl_FProxGD_primal(x0, A, b, mu, opts10);
t10 = toc;

% Augmented Lagrangian Method for Dual Problem
opts11 = []; % modify options
tic;
[x11, iter11, out11] = gl_ALM_dual(x0, A, b, mu, opts11);
t11 = toc;

% Alternating Direction Method of Multipliers for the Dual Problem
opts12 = []; % modify options
tic;
[x12, iter12, out12] = gl_ADMM_dual(x0, A, b, mu, opts12);
t12 = toc;

% Alternating Direction Method of Multipliers for the Primal Problem
opts13 = []; % modify options
tic;
[x13, iter13, out13] = gl_ADMM_primal(x0, A, b, mu, opts13);
t13 = toc;

% Momentum Method
opts14 = []; % modify options
tic;
[x14, iter14, out14] = gl_Momentum(x0, A, b, mu, opts14);
t14 = toc;

% Adaptive Gradient Method
opts15 = []; % modify options
tic;
[x15, iter15, out15] = gl_AdaGrad(x0, A, b, mu, opts15);
t15 = toc;

% RMSProp Method
opts16 = []; % modify options
tic;
[x16, iter16, out16] = gl_RMSProp(x0, A, b, mu, opts16);
t16 = toc;

% Adaptive Momentum Estimation Method
opts17 = []; % modify options
tic;
[x17, iter17, out17] = gl_Adam(x0, A, b, mu, opts17);
t17 = toc;

% print comparison results with cvx-call-mosek
fprintf('     CVX-Mosek: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t1, iter1, out1, sparisity(x1), errfun_exact(x1), errfun(x1, x1), errfun(x2, x1));
fprintf('    CVX-Gurobi: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t2, iter2, out2, sparisity(x2), errfun_exact(x2), errfun(x1, x2), errfun(x2, x2));
fprintf('         Mosek: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t3, iter3, out3, sparisity(x3), errfun_exact(x3), errfun(x1, x3), errfun(x2, x3));
fprintf('        Gurobi: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t4, iter4, out4, sparisity(x4), errfun_exact(x4), errfun(x1, x4), errfun(x2, x4));
fprintf('    PGD Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t5, iter5, out5, sparisity(x5), errfun_exact(x5), errfun(x1, x5), errfun(x2, x5));
fprintf('    SGD Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t6, iter6, out6, sparisity(x6), errfun_exact(x6), errfun(x1, x6), errfun(x2, x6));
fprintf('     GD Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t7, iter7, out7, sparisity(x7), errfun_exact(x7), errfun(x1, x7), errfun(x2, x7));
fprintf('    FGD Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t8, iter8, out8, sparisity(x8), errfun_exact(x8), errfun(x1, x8), errfun(x2, x8));
fprintf(' ProxGD Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t9, iter9, out9, sparisity(x9), errfun_exact(x9), errfun(x1, x9), errfun(x2, x9));
fprintf('FProxGD Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t10, iter10, out10, sparisity(x10), errfun_exact(x10), errfun(x1, x10), errfun(x2, x10));
fprintf('      ALM Dual: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t10, iter11, out11, sparisity(x11), errfun_exact(x11), errfun(x1, x11), errfun(x2, x11));
fprintf('     ADMM Dual: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t10, iter12, out12, sparisity(x12), errfun_exact(x12), errfun(x1, x12), errfun(x2, x12));
fprintf('   ADMM Primal: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t10, iter13, out13, sparisity(x13), errfun_exact(x13), errfun(x1, x13), errfun(x2, x13));
fprintf('      Momentum: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t10, iter13, out13, sparisity(x14), errfun_exact(x14), errfun(x1, x14), errfun(x2, x14));
fprintf('       AdaGrad: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t10, iter13, out13, sparisity(x15), errfun_exact(x15), errfun(x1, x15), errfun(x2, x15));
fprintf('       RMSProp: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t10, iter13, out13, sparisity(x16), errfun_exact(x16), errfun(x1, x16), errfun(x2, x16));
fprintf('          Adam: cpu: %5.2f, iter: %5d, optval: %6.5E, sparisity: %4.3f, err-to-exact: %3.2E, err-to-cvx-mosek: %3.2E, err-to-cvx-gurobi: %3.2E.\n', t10, iter13, out13, sparisity(x17), errfun_exact(x17), errfun(x1, x17), errfun(x2, x17));

plot_results(u, 'Exact', '../figures/gl_exact.png', u, x1, x2)
plot_results(x1, 'CVX-Mosek', '../figures/gl_cvx_mosek.png', u, x1, x2)
plot_results(x2, 'CVX-Gurobi', '../figures/gl_cvx_gurobi.png', u, x1, x2)

plot_results(x3, 'Mosek', '../figures/gl_mosek.png', u, x1, x2)
plot_results(x4, 'Gurobi', '../figures/gl_gurobi.png', u, x1, x2)

% plot_results(x5, 'PGD Primal', '../figures/gl_PGD_primal.png', u, x1, x2)
% plot_results(x6, 'SGD Primal', '../figures/gl_SGD_primal.png', u, x1, x2)

plot_results(x7, 'GD Primal', '../figures/gl_GD_primal.png', u, x1, x2)
plot_results(x8, 'FGD Primal', '../figures/gl_FGD_primal.png', u, x1, x2)

plot_results(x9, 'ProxGD Primal', '../figures/gl_ProxGD_primal.png', u, x1, x2)
plot_results(x10, 'FProxGD Primal', '../figures/gl_FProxGD_primal.png', u, x1, x2)

plot_results(x11, 'ALM Dual', '../figures/gl_ALM_dual.png', u, x1, x2)
plot_results(x12, 'ADMM Dual', '../figures/gl_ADMM_dual.png', u, x1, x2)
plot_results(x13, 'ADMM Primal', '../figures/gl_ADMM_primal.png', u, x1, x2)

plot_results(x14, 'Momentum', '../figures/gl_Momentum.png', u, x1, x2)
plot_results(x15, 'AdaGrad', '../figures/gl_AdaGrad.png', u, x1, x2)
plot_results(x16, 'RMSProp', '../figures/gl_RMSProp.png', u, x1, x2)
plot_results(x17, 'Adam', '../figures/gl_Adam.png', u, x1, x2)
