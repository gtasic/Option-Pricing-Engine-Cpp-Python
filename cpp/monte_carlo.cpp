#include "total.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>





double monte_carlo_call(const MC_parametres& params) {
    std::vector<double> payoffs(params.nb_paths) ;
    std::mt19937 generator(std::random_device{}()) ; // Générateur de nombres aléatoires
    std::normal_distribution<> dist(0.0, 1.0) ; // Distribution normale standard
    for (int i = 0; i< params.nb_paths; i++) {
        double Z = dist(generator) ; // Tirage d'un nombre aléatoire selon une loi normale centrée réduite 
        double ST = params.S0 * exp((params.r - 0.5 * params.sigma * params.sigma) * params.T + params.sigma * sqrt(params.T) * Z) ;
        payoffs[i] = std::max(ST - params.K, 0.0) ;

    }
    double moyenne_payoff = 0.0 ;
    moyenne_payoff = std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / params.nb_paths ;
    double prix_option = exp(-params.r * params.T) * moyenne_payoff ;
    return prix_option ;
}

