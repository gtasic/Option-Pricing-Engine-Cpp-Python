#pragma once
#include <cmath>

struct MC_parametres {
    int nb_simulations;
    int nb_paths ; 
    double S0;    // Prix initial de l'actif sous-jacent
    double K;     // Prix d'exercice de l'option
    double T;     // Temps jusqu'à l'échéance en années
    double r;     // Taux d'intérêt sans risque
    double sigma; // Volatilité de l'actif sous-jacent
} ;

struct tree_parametres {
    double S0;    // Prix initial de l'actif sous-jacent
    double K;     // Prix d'exercice de l'option
    double T;     // Temps jusqu'à l'échéance en années
    double r;     // Taux d'intérêt sans risque
    double sigma; // Volatilité de l'actif sous-jacent
    int N ;       // Nombre de pas dans le modèle binomial
} ;
struct BS_parametres {
    double S0;    // Prix initial de l'actif sous-jacent
    double K;     // Prix d'exercice de l'option
    double T;     // Temps jusqu'à l'échéance en années
    double r;     // Taux d'intérêt sans risque
    double sigma; // Volatilité de l'actif sous-jacent
};

struct D {
    double d1;
    double d2;
};

double inv_sqrt_2pi() ;
double norm_pdf(double x) ;
double norm_cdf(double x) ;

double monte_carlo_call(const MC_parametres& params) ;
double tree(tree_parametres params) ;
double call_price(BS_parametres params) ;
double put_price(BS_parametres params) ;
double call_delta(BS_parametres params) ;
double call_gamma(BS_parametres params) ;
double call_vega(BS_parametres params) ;
double call_theta(BS_parametres params) ;
double call_rho(BS_parametres params) ;
double put_delta(BS_parametres params) ;
double put_gamma(BS_parametres params) ;
double put_vega(BS_parametres params) ;
double put_theta(BS_parametres params) ;
double put_rho(BS_parametres params) ;
double theta_per_day(double theta_annual) ;
double vega_per_1pct(double vega_per_1_0) ; 
