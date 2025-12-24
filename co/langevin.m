
% Langevin Dynamics for Combinatorial Optimization:
% [1] [Fractional Langevin Dynamics for Combinatorial Optimization via Polynomial-Time Escape](https://openreview.net/pdf?id=BZ0igCEeoU)
% [2] [Regularized Langevin Dynamics for Combinatorial Optimization](https://arxiv.org/pdf/2502.00277)

sigmoid = @(x) reshape(1 ./ (1 + exp(-x)), size(x));
kldiv = @(p,q) sum(p.*log2(p./q), 'omitnan');

n = 3;
N = 2^n;

rng default

% convert max-cut graph to qubo
% A = randn(N);
% A = (A+A')/2;
% A(1:N+1:end) = 0;
% qb = maxcut2qubo(graph(A));
% build Q matrix (not PSD!)
% Q = qb.QuadraticTerm;
% Q(1:N+1:end) = qb.LinearTerm;
% H(exact.BestX, Q) == exact.BestFunctionValue

% Indefinite
% Q = randn(n);
% Q = tril(Q,-1) + triu(Q',1);

% PSD
Q = randn(n);
Q = Q + Q';

% Boltzmann
X = (2*(dec2bin(0:N-1, n)=='1') - 1)';
Eb = -(sum(X.*(Q*X), 1)/2);

T = 50;
K = 1e3;

% temperature schedule
% tau = @(t) 0.01 + (1 - t/T);
tau0 = 5;
tau = @(t) tau0/t;
% learning rate schedule
eta = @(t) 1/t;

alpha = 2;
c_alpha = 0.5;

sz = [n 1];
% x in {0,1}
x = randi([0 1], sz);
x2 = x;
% relaxation h in [0 1]
h1 = x; % Gaussian
h2 = x; % Fractional
Et = zeros(2,T);
for t = 1:T

    temp = tau(t);
    lr = eta(t);
    
    % TODO
    % Pt = kernel(Q, temp, lr);
    % mc = dtmc(Pt);
    % [probm, tMix] = asymptotics(mc);
    
    counts = zeros(1, N);
    for k = 1:K
        
        % Vanilla LD
        g1 = dH(2*h1-1, Q);
        h1 = h1 - lr*g1/temp + sqrt(2*lr)*randn(sz);
        % h1 = clip(h1, 0, 1);
        x1 = double(rand(sz) < h1);
        idx = bin2dec(join(string(x1'),""))+1;
        counts(idx) = counts(idx)+1;
    
        % Fractional LD
        g2 = dH(x2, Q);
        
        pg = sigmoid( (0.5*(2*x2 - 1).*g2)/temp );
        Ig = double(rand(sz) < pg); % = binornd(?,pg);
        pz = sigmoid( -(0.5*(2*x2 - 1).*g2)/temp );
        Iz = double(rand(sz) < pz);

        grad_iter = lr*c_alpha*g2/temp;
        z_iter = lr^(1/alpha)*levy(alpha,sz);

        h2 = h2 - Ig.*grad_iter + Iz.*z_iter;
        h2 = clip(h2, 0, 1);
        x2 = double(rand(sz) < h2);
    end
    probl = counts/sum(counts);
    
    Et(1,t) = H(2*x1-1,Q);
    Et(2,t) = H(2*x2-1,Q);
end

% expE = exp(-Eb./temp);
% probb = expE/sum(expE);

figure
% subplot(2,1,1)
hold on
plot(1:T, Et, 'o')
legend('Vanilla', 'Fractional')
yline(min(Eb))
grid on




function z = levy(alpha, sz)
w = pi*rand(sz) - pi/2;
u = rand(sz);
z = (sin(alpha.*w) ./ cos(w).^(1/alpha)) .* (cos((1-alpha).*w) ./ -log(u)).^((1-alpha)/alpha);
end
function e = H(x,Q)
    e = -(x'*Q*x)/2;
end
function e = dH(x,Q)
    e = -Q*x;
end


