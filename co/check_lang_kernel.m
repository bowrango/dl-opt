
% https://arxiv.org/pdf/2206.09914

n = 3;
N = 2^n;

% PSD
Q = randn(n);
Q = Q + Q';

temp = 1;
lr = 0.1;

P = kernel(Q, temp, lr);
mc = dtmc(P);
probm = asymptotics(mc);

X = 2*(dec2bin(0:N-1, n) - '0') - 1;

spin2idx = @(s) bin2dec( num2str((s'+1)/2, '%1d') ) + 1;

K = 1e6;

rng default
x = X(randi(N), :)';
counts = zeros(1,N);
for k = 1:K
    % Vanilla LD (DULA)
    g = dH(x, Q);
    x = x - (lr/temp)*g + sqrt(2*lr)*randn(n,1);
    x = sign(x);
    x(x == 0) = 1;

    idx = spin2idx(x);
    counts(idx) = counts(idx) + 1;
end
probl = counts / sum(counts);

l1 = sum(abs(probm - probl))

function e = dH(x,Q)
    e = -Q*x;
end
function P = kernel(Q, temp, lr)
    n = size(Q,1);
    N = 2^n;
    X = 2*(dec2bin(0:N-1, n) - '0') - 1;
    P = zeros(N,N);
    sigma = sqrt(2*lr);
    for i = 1:N
        x = X(i,:)';          % current spin config (column)
        g = dH(x, Q);         % gradient ∇H(x)
        mu = x - (lr/temp)*g; % mean of Gaussian proposal y

        % Componentwise probabilities P(x'_k = +1 | x)
        p_plus = normcdf(mu / sigma);   % n×1 vector

        for j = 1:N
            s = X(j,:)';      % candidate next spin config

            % For each bit k:
            % if s_k = +1 → use p_plus(k)
            % if s_k = -1 → use 1 - p_plus(k)
            prob = prod( p_plus(s==1) ) * prod( (1 - p_plus(s==-1)) );

            P(i,j) = prob;
        end
    end
end