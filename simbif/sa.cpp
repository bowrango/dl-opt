#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>

// Simulated Annealing for Ising model minimization

/*
Compute Ising energy: H = s^T * J * s
*/
double compute_energy(const std::vector<double>& s, const std::vector<std::vector<double>>& J) {
    int N = s.size();
    double energy = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            energy += s[i] * J[i][j] * s[j];
        }
    }
    return energy;
}

/*
Compute energy change for flipping spin i: ΔE = -2 * s_i * Σ_j J_ij * s_j
Note: Factor of 2 because we count both s_i * J_ij * s_j and s_j * J_ji * s_i
*/
double delta_energy(const std::vector<double>& s, const std::vector<std::vector<double>>& J, int i) {
    int N = s.size();
    double delta = 0.0;
    for (int j = 0; j < N; ++j) {
        delta += J[i][j] * s[j];
    }
    return -2.0 * s[i] * delta;
}

/*
Simulated Annealing: probabilistic local search with thermal fluctuations
- Explores via single spin flips, accepting worse moves with probability exp(-ΔE/T)
- Temperature T decreases geometrically from T_initial to T_final
- Multiple restarts (K batches) for better exploration
*/
std::vector<std::vector<double>> SimulatedAnnealing(const std::vector<std::vector<double>>& J,
                                                     int K = 10,
                                                     int steps = 10000,
                                                     double T_initial = 2.0,
                                                     double T_final = 0.01) {
    int N = J.size();
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> spin_dist(0, N - 1);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::uniform_int_distribution<int> init_spin_dist(0, 1);

    std::vector<std::vector<double>> results(K, std::vector<double>(N));

    // Geometric cooling schedule
    double alpha = std::pow(T_final / T_initial, 1.0 / steps);

    // Run K independent annealing trajectories
    for (int k = 0; k < K; ++k) {
        // Initialize with random spins
        std::vector<double> s(N);
        for (int i = 0; i < N; ++i) {
            s[i] = init_spin_dist(gen) == 0 ? -1.0 : 1.0;
        }

        double T = T_initial;

        // Annealing loop
        for (int step = 0; step < steps; ++step) {
            // Propose random spin flip
            int i = spin_dist(gen);
            double dE = delta_energy(s, J, i);

            // Metropolis acceptance criterion
            if (dE <= 0 || uniform_dist(gen) < std::exp(-dE / T)) {
                s[i] = -s[i];
            }

            // Cool down
            T *= alpha;
        }

        results[k] = s;
    }

    return results;
}

#ifndef SA_NO_MAIN
int main() {
    // This is a standalone implementation - see bench.cpp for comparison
    std::cout << "Simulated Annealing implementation for Ising models" << std::endl;
    std::cout << "Use bench.cpp to compare against Simulated Bifurcation" << std::endl;
    return 0;
}
#endif
