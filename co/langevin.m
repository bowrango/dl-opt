
% Langevin Dynamics for Combinatorial Optimization:
% [1] [Fractional Langevin Dynamics for Combinatorial Optimization via Polynomial-Time Escape](https://openreview.net/pdf?id=BZ0igCEeoU)
% [2] [Regularized Langevin Dynamics for Combinatorial Optimization](https://arxiv.org/pdf/2502.00277)

N = 4;

rng default

% convert max-cut graph to qubo
A = randn(N);
A = (A+A')/2;
A(1:N+1:end) = 0;
qb = maxcut2qubo(graph(A));

% exact = solve(qb);

% build Q matrix (not PSD!)
Q = qb.QuadraticTerm;
Q(1:N+1:end) = qb.LinearTerm;

% H(exact.BestX, Q) == exact.BestFunctionValue

sigmoid = @(x) 1 / (1 + exp(-x));

dt = 0.01;
T = 100;
Nsteps = T / dt;

% temperature schedule
tau = @(t) 0.01 + (1 - t/T);

sz = [N 1];
x = randi([0 1], sz); % {0,1}
h = x; % relaxation [0 1]
E = zeros(1,T);
for t = 1:T
    temp = tau(t);

    % pg = sigmoid( -dH(x,Q)/temp ).';
    % Ig = double(rand(sz) < pg); % = binornd(?,pg);
    % pz = sigmoid( -dH(x,Q)/temp ).';
    % Iz = double(rand(sz) < pz);
    % grad_iter = dt*(-1/temp)*dH(x,Q);
    % z_iter = sqrt(dt)*randn(sz);
    % update
    % h = h - Ig.*grad_iter + Iz.*z_iter;

    drift = -dH(h, Q)/temp;
    noise = sqrt(2*dt)*randn(sz);
    
    h = h + drift*dt + noise;
    h = clip(h, 0, 1);

    x = double(rand(sz) < h);
    E(t) = H(x,Q);
end

plot(1:T, E)


% drift/score and diffusion
% F = @(t, x) -dH(x,Q)/tau(t);
% F = @(t, z) drift(t, z, Q, tau);
% G = @(t, x) sqrt(2).*eye(N);
% 
% model = sde(F, G, StartState=x);
% [U, T] = simulate(model, Nsteps);
% 
% Z = sigmoid(U);
% 
% X = Z > rand(size(Z));
% 
% plot(T, X(:,1))



function e = H(x,Q)
    e = x'*Q*x;
end
function e = dH(x,Q)
    e = 2*Q*x;
end

% function du = drift(t, h, Q, tau)
%     % Drift in logit-space
%     dsigmoid = @(h) h .* (1 - h);
% 
%     h = (1 / (1 + exp(-h))).';
%     x = h;
%     du = -(dH(x,Q) .* dsigmoid(h))/tau(t);
% end

