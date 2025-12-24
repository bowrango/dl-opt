
% Simulated annealing with pbits via Gibbs sampling

kldiv = @(p,q) sum(p.*log2(p./q), 'omitnan');

n = 8;
N = 2^n;

tau0 = 5;
tau = @(t) tau0/t;

rng default

% symmetric i.i.d Gaussian Qij ~ N(0, 1)
Q = randn(n);
h = diag(Q);
Q = tril(Q,-1) + triu(Q',1);

T = 20;

% Boltzmann
% E(k) = -(0.5*X(:,k)'*Q*X(:,k) + h'*X(:,k));
X = (2*(dec2bin(0:N-1, n)=='1') - 1)';
Eb = -(sum(X.*(Q*X), 1)/2 + h'*X);
[Eb, perm] = sort(Eb);

numGibbsSteps = 1e3;

pbits = sign(randn(n, 1));
Et = zeros(numGibbsSteps,T);
F = zeros(N, T);
kl = zeros(1,T);
for t = 1:T
    beta = 1/tau(t);

    % Markov Transition
    Pt = unitrans(Q, h, beta);
    mc = dtmc(Pt);
    [probm, tMix] = asymptotics(mc);
    probm = probm';
    probm = probm(perm);
    
    % Gibbs Sampling
    counts = zeros(N, 1);
    for k = 1:numGibbsSteps
        idx = randi(n); % random-scan
   
        I = Q(idx,:)*pbits + h(idx);
        pbits(idx) = sign(tanh(beta*I) - (2*rand-1));
    
        bits = (pbits+1)/2;
        idx = bin2dec(join(string(bits'),""))+1;
        counts(idx) = counts(idx)+1;
        Et(k,t) = -(0.5*(pbits'*Q*pbits) + h'*pbits);
    end
    probg = counts/sum(counts);
    probg = probg(perm);
    
    kl(t) = kldiv(probg, probm);

    F(:,t) = probg;
end

% aggregate probability by energy
[Ebin, ~, eidx] = unique(Eb);
Fbin = zeros(length(Ebin), T);
for t = 1:T
    for k = 1:length(Ebin)
        Fbin(k,t) = sum(F(eidx==k, t));
    end
end

% remove near-zero probability
minProb = 1/numGibbsSteps;
mask = any(Fbin >= minProb, 2);
Eplot = Ebin(mask);
Fplot = Fbin(mask,:);

[Tgrid, Egrid] = meshgrid(1:T, Eplot);
mask = Fplot >= minProb;

figure
subplot(2,1,1)
hold on
scatter(Tgrid(mask), Egrid(mask), 200, Fplot(mask), 'filled');
yline(min(Eb))
colormap(sky)
cbar = colorbar; 
ylabel(cbar,'Probability')
ylabel('Energy')
grid on

subplot(2,1,2)
plot(1:T, kl)
grid on
ylabel('KL(p_t,p_{tau})', Interpreter='tex')
xlabel('Time')

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