#include "total.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>



double tree(tree_parametres params) {
    double delta_t = params.T / params.N ;
    double u = std::exp(params.sigma * std::sqrt(delta_t)) ;
    double d = 1 / u ;
    double p = (std::exp(params.r * delta_t) - d) / (u - d) ;
    std::vector<std::vector<double>> stock_price(params.N + 1, std::vector<double>(params.N + 1)) ;  //on crée un tableau carré de double de taille N+1 pour stocker le prix d'une option au range N+1 
    for (int i = 0 ; i <= params.N ; i++) {
        for (int j = 0 ; j <= i ; j++) {
            stock_price[j][i] = params.S0 * std::pow(u, i - j) * std::pow(d, j) ;  // On regarde le prix de l'action à chaque noeud
        }
    }
    std::vector<double> valeur(params.N+1) ;   //on crée un tableau qui rasemble les valeurs aux derniers noeuds
    for (int k = 0 ; k< params.N+1; k++) {
        valeur[k] = stock_price[k][params.N] ;
    }
    std::vector<double> valeur_base(params.N+1) ;    //on crée un tableau qui rasemble l'améliration au dernier noeud 
    for (int k = 0 ; k< params.N+1; k++) {
        valeur_base[k] = std::max(valeur[k]-params.K,0.0);
    }
    std::vector<std::vector<double>> valeur_finale(params.N+1, std::vector<double>(params.N+1)) ;// tableau de N+1 par N+1 pour stocker les valeurs de l'option à chaque noeud
    for (int i = 0 ; i <= params.N ; i++) {
        valeur_finale[i][params.N] = valeur_base[i] ; // On initialise la dernière colonne du tableau des valeurs de l'option avec les valeurs de l'option au dernier noeud
    }
    for (int j = params.N-1 ; j >= 0 ; j--) {
        for (int k = 0 ; k <= j ; k++) {
            valeur_finale[k][j] = std::exp(-params.r * delta_t) * (p * valeur_finale[k][j + 1] + (1 - p) * valeur_finale[k + 1][j + 1]) ;
        }
    }
    return valeur_finale[0][0] ;
}
/*
int main() {
    tree_parametres params{100.0, 100.0, 1.0, 0.05, 0.2, 100} ;
    double prix_option = tree(params) ;
    std::cout << "Le prix de l'option est : " << prix_option << std::endl ;
    return 0 ;
}
*/