
% Stationary p-bit distribution

n = 10;
N = 2^n;
beta = 0.1;

rng default

% symmetric i.i.d Gaussian Qij ~ N(0, 1)
Q = randn(n);
h = diag(Q);
Q = tril(Q,-1) + triu(Q',1);

% 1. Markov Transition
T1 = unitrans(Q, h, beta);
mc = dtmc(T1);
[probm, tMix] = asymptotics(mc);
probm = probm';   
[ev, lambda] = eigs(T1.', 2);
sgap = 1 - lambda(2);
epsilon = 1e-3;
tmix = log(1/epsilon) / sgap;
numGibbsSteps = 1e6;

% 2. Gibbs Sampling
pbits = sign(randn(n, 1));
counts = zeros(N, 1);
for t = 1:numGibbsSteps
    idx = randi(n); % random-scan
    I = Q(idx,:)*pbits + h(idx);
    pbits(idx) = sign(tanh(beta*I) - (2*rand-1));
    bits = (pbits+1)/2;
    idx = bin2dec(join(string(bits'),""))+1;
    counts(idx) = counts(idx) + 1;
end
probg = counts/sum(counts);

% 3. Boltzmann
% E(k) = -(0.5*X(:,k)'*Q*X(:,k) + h'*X(:,k));
X = (2*(dec2bin(0:N-1, n)=='1') - 1)';
E = -(sum(X.*(Q*X), 1)/2 + h'*X);
expE = exp(-beta.*E);
probb = expE/sum(expE);

[E, perm] = sort(E);
probg = probg(perm);
probb = probb(perm);
probm = probm(perm);

figure
grid on
hold on
plot(E, probg, 'bx')
plot(E, probb, 'ro')
plot(E, probm, 'r-')
xlabel('Energy H(x)')
ylabel('Probability')
legend('Gibbs', 'Boltzmann', 'Markov', Location='best')

function P = unitrans(Q, h, beta)
% Transition matrix for a uniform random p-bit update (random scan)
% P(i,j) is the probability state i transitions to state j.
% Rows sum to 1. States with Hamming distance more than 1 have no
% probability.
n = size(Q,1);
N = 2^n;
X = 2*(dec2bin(0:N-1, n)=='1') - 1;
P = zeros(N);
for ii = 1:N
    x = X(ii, :);
    for idx = 1:n
        jj = bitxor(ii-1, bitshift(1, n-idx)) + 1;

        f = (Q(idx,:)*x' + h(idx));
        % p = (1 - x(idx)*tanh(beta*f))/2;    
        p = 1 / (1 + exp(2*beta*x(idx)*f));
        
        P(ii,jj) = P(ii,jj) + p/n;
        P(ii,ii) = P(ii,ii) + (1-p)/n;
    end
end
end