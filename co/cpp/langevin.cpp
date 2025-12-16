#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>

// Langevin Dynamics for Combinatorial Optimization:
// [1] [Fractional Langevin Dynamics for Combinatorial Optimization via Polynomial-Time Escape](https://openreview.net/pdf?id=BZ0igCEeoU)
// [2] [Regularized Langevin Dynamics for Combinatorial Optimization](https://arxiv.org/pdf/2502.00277)

// g++ langevin.cpp -o langevin -std=c++17

struct MaxCut {
    std::vector<std::vector<double>> W;

    explicit MaxCut(size_t n) : W(n, std::vector<double>(n, 0.0)) {}

    size_t dim() const { return W.size(); }

    void set_edge(size_t i, size_t j, double w) {
        if (i == j) return; // no self-loops
        W[i][j] = W[j][i] = w;
    }

    double eval(const std::vector<double>& x) const {
        double H = 0.0;
        size_t n = dim();
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double wij = W[i][j];
                if (wij == 0.0) continue;
                H += wij * (x[i] * x[j] - 0.5 * x[i] - 0.5 * x[j]);
            }
        }
        return H;
    }

    void grad(const std::vector<double>& x, std::vector<double>& g) const {
        size_t n = dim();
        g.assign(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double wij = W[i][j];
                if (wij == 0.0) continue;
                g[i] += wij * (x[j] - 0.5);
                g[j] += wij * (x[i] - 0.5);
            }
        }
    }
};

// FIXME
struct QUBO {
    // Combinatorial Objective Penalty Function
    // H(x) = a(x) + /lambda*b(x)
    std::vector<double> w;
    double lambda;

    QUBO(size_t dim, double lambda_ = 0.1) : w(dim, 0.0), lambda(lambda_) {
        for (size_t i = 0; i < dim; ++i) {w[i] = static_cast<double>(i + 1);}
    }

    double eval(const std::vector<double>& x) const {
        double H = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            H += w[i] * x[i] + lambda * x[i] * x[i];
        }
        return H;
    }

    void grad(const std::vector<double>& x, std::vector<double>& g) const {
        g.resize(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            g[i] = w[i] + 2.0 * lambda * x[i];
        }
    }
};

double sample_sas(double alpha, std::mt19937_64& rng) {
    // Theorem 3 with \beta = 0 (symmetric \alpha-stable)
    // U ~ Uniform(-pi/2, pi/2), W ~ Exp(1).
    std::uniform_real_distribution<double> uni(-M_PI / 2.0, M_PI / 2.0);
    std::exponential_distribution<double> expo(1.0);

    double U = uni(rng);
    double W = expo(rng);
    if (std::abs(alpha - 1.0) > 1e-8) {
        double denom_cos = std::pow(std::cos(U), 1.0 / alpha);
        double inner = std::cos(U - alpha * U) / W;
        double pow_term = std::pow(inner, (1.0 - alpha) / alpha);
        return std::sin(alpha * U) / denom_cos * pow_term;
    } else {
        return (M_PI / 2.0) * std::tan(U);
    }
}

struct FLDSampler {
    // Fractional Langevin Dynamics Sampler
    QUBO& energy;
    double alpha;     // stability parameter in (0,2]
    double tau;       // "temperature"
    double eta;       // step size
    double c_alpha;   // scale factor from the paper
    std::mt19937_64 rng;

    FLDSampler(QUBO& e,
               double alpha_ = 1.5,
               double tau_   = 1.0,
               double eta_   = 0.01,
               uint64_t seed = 42)
        : energy(e), alpha(alpha_), tau(tau_), eta(eta_), rng(seed)
    {
        // c_alpha = Γ(α - 1) / Γ(α/2)^2   (from the paper)
        // Note: this is for α > 1; for α <= 1 you may adjust or treat as a tunable constant.
        if (alpha > 1.0) {
            double num  = std::tgamma(alpha - 1.0);
            double den  = std::tgamma(alpha / 2.0);
            c_alpha = num / (den * den);
        } else {
            // For α <= 1, use a reasonable constant (can be tuned)
            c_alpha = 1.0;
        }
    }

    void step(std::vector<double>& x) {
        // Equation 19 [1]
        std::vector<double> g;
        energy.grad(x, g);

        double drift_scale = eta * c_alpha / tau;
        double noise_scale = std::pow(eta, 1.0 / alpha);
        for (size_t i = 0; i < x.size(); ++i) {
            // Drift
            x[i] -= drift_scale * g[i];
            // Levy Noise
            x[i] += noise_scale * sample_sas(alpha, rng);
        }
    }
};

int main() {
    const size_t dim = 256;
    QUBO f(dim, 0.1);

    // α = 2.0 -> LD (Gaussian)
    // α < 2 -> FLD (heavy-tailed)
    double alpha = 1.5;
    double tau   = 1.0;
    double eta   = 0.01;

    FLDSampler sampler(f, alpha, tau, eta, 1234);

    std::mt19937_64 rng_init(5678);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    std::vector<double> x(dim);
    for (auto &xi : x) {xi = uni01(rng_init);}

    std::cout << "Initial f(x) = " << f.eval(x) << std::endl;

    const int num_steps = 1000;
    for (int t = 0; t < num_steps; ++t) {
        sampler.step(x);
        if ((t + 1) % 1000 == 0) {
            std::cout << "Step " << (t + 1)
                      << ", f(x) = " << f.eval(x) << std::endl;
        }
    }

    // std::vector<int> x_binary(dim);
    // for (size_t i = 0; i < dim; ++i) {
    //     x_binary[i] = (x[i] > 0.5) ? 1 : 0;
    // }

    // std::cout << "Final relaxed x:\n";
    for (double xi : x) std::cout << xi << " ";
    // std::cout << "\nThresholded x (CO solution approximation):\n";
    // for (int xi : x_binary) std::cout << xi << " ";
    // std::cout << std::endl;

    return 0;
}