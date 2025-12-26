#include "total.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>



double tree(tree_parametres params) {
    auto t0 = std::chrono::high_resolution_clock::now();
    
    double delta_t = params.T / params.N;
    double u = std::exp(params.sigma * std::sqrt(delta_t));
    double d = 1.0 / u;
    double p = (std::exp(params.r * delta_t) - d) / (u - d);
    double df = std::exp(-params.r * delta_t); // Discount factor par pas

    // OPTIMISATION : On n'alloue qu'un seul vecteur de taille N+1
    // Au lieu de stocker tout l'arbre, on stocke juste les valeurs actuelles
    std::vector<double> values(params.N + 1);

    // 1. Initialisation aux feuilles (Maturité)
    // ST à la feuille j correspond à S0 * u^(N-j) * d^j
    for (int j = 0; j <= params.N; ++j) {
        double ST = params.S0 * std::pow(u, params.N - j) * std::pow(d, j);
        values[j] = std::max(ST - params.K, 0.0); // Payoff Call
    }

    // 2. Remontée de l'arbre (Backward Induction)
    // On écrase les valeurs du vecteur 'values' étape par étape
    for (int i = params.N - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            // C'est ici que la magie opère : values[j] devient la valeur à l'étape i
            // en utilisant les valeurs de l'étape i+1 (qui sont actuellement dans values[j] et values[j+1])
            values[j] = df * (p * values[j] + (1.0 - p) * values[j + 1]);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration<double>(t1 - t0).count();
    
    // Le prix est maintenant en values[0], mais vous retournez le runtime
    return values[0]; 
}
/*
int main() {
    std::vector<int> Nu = {10, 25, 50, 100, 200, 500, 750, 1000, 1500, 3000, 5000, 10000, 20000};
    
    std::cout << "N, Runtime(s)" << std::endl;
    
    for (int i = 0; i < Nu.size(); i++) { // CORRECTION: int i = 0
        tree_parametres params;
        params.N = Nu[i];
        params.S0 = 100.0;
        params.K = 110.0;
        params.T = 0.2;
        params.r = 0.01;
        params.sigma = 0.3;
        
        double time = tree(params);
        std::cout << params.N << ", " << time << std::endl;
    }

    return 0;
}
*/