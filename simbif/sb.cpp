#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

// Simulated Bifurcation solves Ising problems via adiabatic bifurcation in Hamiltonian dynamics [1].
// At the first bifurcation point, x approximately becomes the eigenvector of J corresponding to max eigenvalue. 
// This provides an approximate solution of the Ising problem obtained by a continuous reduction method.

// [1]: https://www.science.org/doi/full/10.1126/sciadv.aav2372

std::pair<double, std::vector<double>> calculate_min_J_o(const std::vector<std::vector<double>>& J) {
    /*
    Exact solver for ordered J using two-cluster pattern theorem [1].
    For ordered couplings, the ground state partitions spins into two contiguous clusters.
    This reduces the search space from O(2^N) to O(N).
    */
    int N = J.size();
    std::vector<double> H_l;
    std::vector<std::vector<double>> s_l;

    for (int M = 1; M < N; ++M) {
        std::vector<double> s(N);
        // Set first M elements to 1
        for (int i = 0; i < M; ++i) {
            s[i] = 1.0;
        }
        // Set remaining elements to -1
        for (int i = M; i < N; ++i) {
            s[i] = -1.0;
        }

        double H = 0.0;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                H += s[i] * J[i][j] * s[j];
            }
        }
        H_l.push_back(H);
        s_l.push_back(s);
    }

    // TODO
    auto min_H_iter = std::min_element(H_l.begin(), H_l.end());
    int min_H_index = std::distance(H_l.begin(), min_H_iter);
    return { *min_H_iter, s_l[min_H_index] };
}

std::vector<std::vector<double>> J_o(int N, double d = 1.0) {
    /*
    Ordered Coupling: J_ij = (i/N)^d + (j/N)^d normalized by N^2.
    Structure enables analytical solution via two-cluster theorem.
    */
    std::vector<std::vector<double>> matrix(N, std::vector<double>(N));
    double scale = std::pow(N, 2);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = (std::pow((i + 1.0)/N, d) + std::pow((j + 1.0)/N, d)) / scale;
        }
    }
    return matrix;
}

std::vector<std::vector<double>> randnj(int N, double sigma) {
    /*
    Random Coupling: J_ij ~ Normal(0, sigma^2)
    Largest eigenvalue is about 2*sigma*sqrt(N) by Wigner hemicircle
    */
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, sigma);

    std::vector<std::vector<double>> matrix(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double val = dist(gen);
            matrix[i][j] = val;
            matrix[j][i] = val;
        }
    }
    return matrix;
}

double potential(const std::vector<double>& s, const std::vector<std::vector<double>>& J) {
    // Compute Ising energy: H = -0.5(s^T*J*s)
    int N = s.size();
    double H = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            H += s[i] * J[i][j] * s[j];
        }
    }
    return -0.5*H;
}

std::vector<std::vector<double>> g(const std::vector<std::vector<double>>& x) {
    // Clamp positions to [-1 1]
    std::vector<std::vector<double>> result(x.size(), std::vector<double>(x[0].size()));
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < x[i].size(); ++j) {
            result[i][j] = std::clamp(x[i][j], -1.0, 1.0);
        }
    }
    return result;
}

std::vector<std::vector<double>> h(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y) {
    // TODO Clamp momentum to 0 when position hits the boundary to prevent oscillation
    std::vector<std::vector<double>> result(y.size(), std::vector<double>(y[0].size()));
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < x[i].size(); ++j) {
            result[i][j] = std::abs(x[i][j]) <= 1 ? y[i][j] : 0;
        }
    }
    return result;
}

std::vector<std::vector<double>> simbif(const std::vector<std::vector<double>>& J, int K = 3, int steps = 10000, double eps0 = 1, double p_f = 1, double delta_t = 0.1, const char* filename = nullptr) {
    // Simulated Bifurcation to find the groundstate of J
    /*
    - x: continuous relaxation of discrete spins
    - y: conjugate momenta
    - a: bifurcation parameter
    - K: number of parallel batches
    */
    int N = J.size();

    // TODO
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Initialize K batches near origin (symmetric bifurcation point)
    std::vector<std::vector<double>> x(K, std::vector<double>(N));
    std::vector<std::vector<double>> y(K, std::vector<double>(N));
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            x[i][j] = dist(gen);
            y[i][j] = dist(gen);
        }
    }

    // Bifurcation parameter: p=0 (symmetric), p=1 (bifurcated)
    double p = 0;

    std::ofstream readout;
    if (filename != nullptr) {
        readout.open(filename);
        readout << "# step\tmin_energy\n";
    }

    // Simulate bifurcation with modified explicit symplectic Euler
    for (int step = 0; step < steps; ++step) {

        // Compute interaction forces: f = x * J
        std::vector<std::vector<double>> f(K, std::vector<double>(N, 0.0));
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < N; ++k) {
                    f[i][j] += x[i][k] * J[k][j];
                }
            }
        }

        // Update momentum (eq. 13): dy/dt = -(p_f-p)x + eps0 * J * x
        // Restoring force (p_f-p)x weakens as p->p_f to allow bifurcation
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < N; ++j) {
                y[i][j] += ((-(p_f - p) * x[i][j] + eps0 * f[i][j]) * delta_t);
            }
        }

        // Update position (eq. 14): dx/dt = p_f * y
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < N; ++j) {
                x[i][j] += (p_f * y[i][j] * delta_t);
            }
        }
        
        // Linearly increase bifurcation
        p += p_f / steps;

        // Clip positions and momentum
        x = g(x);
        y = h(x, y);

        if (filename != nullptr && step % 10 == 0) {
            double min_E = std::numeric_limits<double>::max(); 
            for (int i = 0; i < K; ++i) {
                double E = potential(x[i], J);
                if (E < min_E) {min_E = E;}
            }
            readout << step << "\t" << min_E << "\t" << "\n";
        }
    }

    if (filename != nullptr) {
        readout.close();
    }

    // Project continuous positions to spins {-1, +1}
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            x[i][j] = std::copysign(1.0, x[i][j]);
        }
    }

    return x;
}

#ifndef SB_NO_MAIN
int main() {

    int N = 1000;

    double sigma = 1.0;
    std::vector<std::vector<double>> J = randnj(N, sigma);

    int K = 1;
    int steps = 1000;
    double delta = 1.0;
    // See Supplementary Information
    double eps = 0.5*delta / (sigma * sqrt(N));

    double p_f = 1;
    double delta_t = 0.5;

    // Timestamp results
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "sb_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".dat";
    std::string filename = ss.str();

    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<double>> min_x = simbif(J, K, steps, eps, p_f, delta_t, filename.c_str());
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() / 1000.0 << " seconds.\n";

    // Negate J back for calculate_min_J_o
    // std::vector<std::vector<double>> neg_J = J;
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         neg_J[i][j] = -J[i][j];
    //     }
    // }
    // auto [min_H, true_gs] = calculate_min_J_o(neg_J);

    return 0;
}
#endif