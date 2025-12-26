#include "total.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <chrono>
#include <vector>





double monte_carlo_call(const MC_parametres& params) {
    auto t0 = std::chrono::high_resolution_clock::now();

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
    
    auto t1 = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration<double>(t1 - t0).count();

    return prix_option ;
}
/*
int main(){
    std::vector<int> Nu = {10,25,50,100,200,500,750,1000,1500,3000,5000,10000,20000};
    for (int i;i< Nu.size(); i++){
        MC_parametres params; 
        params.nb_simulations = Nu[i];
        params.nb_paths = Nu[i];
        params.S0 = 100.0;
        params.K = 110.0;
        params.T =    0.2;
        params.r    = 0.01;
        params.sigma = 0.3;
        std::vector<double> result{} ; 
        result.push

    return 0;
}
    */

