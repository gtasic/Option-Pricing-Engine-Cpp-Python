#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <iomanip>
#include "total.hpp"


// Structure des paramètres du modèle
/*
struct HestonParams {
    double S0;    // prix initial du sous-jacent
    double v0;    // variance initiale
    double r;     // taux sans risque (const)
    double kappa; // vitesse de mean-reversion
    double theta; // variance de long terme
    double sigma; // vol de la variance (vol-of-vol)
    double rho;   // corrélation entre S et v
};
*/

RNGWrapper::RNGWrapper(unsigned long long seed)
    : rng(seed), nd(0.0, 1.0)
{

}

double RNGWrapper::next_normal() {
    return nd(rng);
}

double simulate_one_path_call(
    const HestonParams &p,
    double K,
    double T,
    int M,              // nombre de steps
    RNGWrapper &rngw,
    bool antithetic 
   // std::mt19937_64 &rng,
   //std::normal_distribution<double> &nd
) {
    double dt = T / static_cast<double>(M);
    double sqrt_dt = std::sqrt(dt);

    double S = p.S0;
    double v = p.v0;

    for (int i = 0; i < M; ++i) {
        // normals indépendants
        double Z1 = rngw.next_normal();
        double Z3 = rngw.next_normal();

    if (antithetic) { // <── inversion des signes
        Z1 = -Z1;
        Z3 = -Z3;
        }

        // corrélé
        double Z2 = p.rho * Z1 + std::sqrt(1.0 - p.rho * p.rho) * Z3;

        // full-truncation Euler pour v
        double v_pos = std::max(v, 0.0);
        double dv = p.kappa * (p.theta - v_pos) * dt + p.sigma * std::sqrt(v_pos) * sqrt_dt * Z1;
        v = v + dv;
        // optional: enforce small negative cutoff
        if (v < 0.0) v = 0.0;

        // mise à jour de S (log-Euler / exponentielle)
        double drift = (p.r - 0.5 * v_pos) * dt;
        double diffusion = std::sqrt(v_pos) * sqrt_dt * Z2;
        S = S * std::exp(drift + diffusion);
    }

    double payoff = std::max(S - K, 0.0);
    return payoff;
}

// Monte Carlo pricing (calls simulate_one_path_call)
/*
struct MCResult {
    double price;
    double std_error;
    double conf95_low;
    double conf95_high;
    int n_paths;
    double runtime_seconds;
};
*/

MCResult price_european_call_mc(
    const HestonParams &p,
    double K,
    double T,
    int M,
    int N,                     // nombre de chemins
    unsigned long long seed ,
    bool antithetic 
) {
  //  std::mt19937_64 rng(seed);
  //  std::normal_distribution<double> nd(0.0, 1.0);
    RNGWrapper rngw(seed);

    std::vector<double> payoffs;
    payoffs.reserve(N);

    auto t0 = std::chrono::high_resolution_clock::now();

    if (!antithetic) {
        for (int i = 0; i < N; ++i) {
            double payoff = simulate_one_path_call(p, K, T, M, rngw);
            payoffs.push_back(std::exp(-p.r * T) * payoff);
        }
    } else {

        for (int i = 0; i < N; i += 2) {
            double payoff1 = simulate_one_path_call(p, K, T, M, rngw);
         //   std::mt19937_64 local_rng(rng()); 
       //     std::normal_distribution<double> local_nd(0.0, 1.0);

            double payoff2 = simulate_one_path_call(p, K, T, M, rngw, /*antithetic=*/true);

            payoffs.push_back(std::exp(-p.r * T) * payoff1);
            if ((int)payoffs.size() < N) // only push second if needed
                payoffs.push_back(std::exp(-p.r * T) * payoff2);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration<double>(t1 - t0).count();

    double mean = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / payoffs.size();
    double sqsum = 0.0;
    for (double x : payoffs) sqsum += (x - mean) * (x - mean);
    double variance = sqsum / (payoffs.size() - 1);
    double stddev = std::sqrt(variance);
    double stderr = stddev / std::sqrt(static_cast<double>(payoffs.size()));

    double z95 = 1.96;
    double ci_low = mean - z95 * stderr;
    double ci_high = mean + z95 * stderr;

    MCResult res;
    res.price = mean;
    res.std_error = stderr;
    res.conf95_low = ci_low;
    res.conf95_high = ci_high;
    res.n_paths = static_cast<int>(payoffs.size());
    res.runtime_seconds = runtime;
    return res;
}

int main() {
    // Exemple de paramètres
    HestonParams p;
    p.S0 = 100.0;  // le prix du sous-jacent 
    p.v0 = 0.04;  // la variance ini
    p.r = 0.01;// le taux d'inérêt
    p.kappa = 2.0;  // la vitesse de variance
    p.theta = 0.04; //
    p.sigma = 0.3; // la vol
    p.rho = -0.7;  // le greeks rho

    double K = 100.0;
    double T = 1.0;    // maturité en années
    int M = 252;       // steps (journaliers)
    int N = 20000;    // chemins Monte Carlo 
    unsigned long long seed = 123456789ULL;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Heston Monte-Carlo pricing (full-truncation Euler)\n";
    std::cout << "S0=" << p.S0 << " v0=" << p.v0 << " r=" << p.r
              << " kappa=" << p.kappa << " theta=" << p.theta
              << " sigma=" << p.sigma << " rho=" << p.rho << "\n";
    std::cout << "K=" << K << " T=" << T << " M=" << M << " N=" << N << "\n";

    MCResult res = price_european_call_mc(p, K, T, M, N, seed, /*antithetic=*/false);

    std::cout << "Price (MC) = " << res.price << "\n";
    std::cout << "Std error = " << res.std_error << "   (n=" << res.n_paths << ")\n";
    std::cout << "95% CI = [" << res.conf95_low << ", " << res.conf95_high << "]\n";
    std::cout << "Runtime (s) = " << res.runtime_seconds << "\n";

    return 0;
}
