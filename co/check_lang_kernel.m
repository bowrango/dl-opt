
% https://arxiv.org/pdf/2206.09914

n = 3;
N = 2^n;

% PSD
rng default
Q = randn(n);
Q = Q + Q';

temp = 1;
lr = 0.1;

P = kernel(Q, temp, lr);
mc = dtmc(P);
[probm, tMix] = asymptotics(mc);

% X in {-1,+1}
X = 2*(dec2bin(0:N-1, n) - '0') - 1;

spin2idx = @(s) bin2dec( num2str((s'+1)/2, '%1d') ) + 1;

K = 1e5;

s = X(randi(N), :)';
counts = zeros(1,N);
for k = 1:K
    % Vanilla LD (DULA)
    g = dH(s, Q);
    s = sign(s - (lr/temp)*g + sqrt(2*lr)*randn(n,1));
    s(s == 0) = 1;

    idx = spin2idx(s);
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
        x = X(i,:)';
        g = dH(x, Q);
        mu = x - (lr/temp)*g;
        % componentwise probabilities
        p_plus = normcdf(mu / sigma);
        for j = 1:N
            s = X(j,:)';
            prob = prod(p_plus(s==1)) * prod((1 - p_plus(s==-1)));
            P(i,j) = prob;
        end
    end
end