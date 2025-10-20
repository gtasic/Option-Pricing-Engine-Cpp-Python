#include "total.hpp"
#include <cmath>
#include <iostream>

double inv_sqrt_2pi() { return 1.0 / std::sqrt(2.0 * M_PI); }

double norm_pdf(double x) {
    return inv_sqrt_2pi() * std::exp(-0.5 * x * x);
}
double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}


D d1d2(BS_parametres params) {
    const double vsqrtT = params.sigma * std::sqrt(params.T);
    D d{};
    d.d1 = (std::log(params.S0 / params.K) + (params.r + 0.5 * params.sigma * params.sigma) * params.T) / vsqrtT;
    d.d2 = d.d1 - vsqrtT;
    return d;
}

double call_price(BS_parametres params) {
    auto [d1, d2] = d1d2(params);
    return params.S0 * norm_cdf(d1) - params.K * std::exp(-params.r * params.T) * norm_cdf(d2);
}
double put_price(BS_parametres params) {
    auto [d1, d2] = d1d2(params);
    return params.K * std::exp(-params.r * params.T) * norm_cdf(-d2) - params.S0 * norm_cdf(-d1);
}


double call_delta(BS_parametres params) {
    return norm_cdf(d1d2(params).d1);
}
double call_gamma(BS_parametres params) {
    auto [d1, _] = d1d2(params);
    return norm_pdf(d1) / (params.S0 * params.sigma * std::sqrt(params.T));
}
double call_vega(BS_parametres params) {
    auto [d1, _] = d1d2(params);
    return params.S0 * norm_pdf(d1) * std::sqrt(params.T); // par 1.0 de sigma
}
double call_theta(BS_parametres params) {
    auto [d1, d2] = d1d2(params);
    double term1 = - (params.S0 * norm_pdf(d1) * params.sigma) / (2.0 * std::sqrt(params.T));
    double term2 = - params.r * params.K * std::exp(-params.r * params.T) * norm_cdf(d2);
    return term1 + term2; // par an
}
double call_rho(BS_parametres params) {
    auto [_, d2] = d1d2(params);
    return params.K * params.T * std::exp(-params.r * params.T) * norm_cdf(d2);
}


double put_delta(BS_parametres params) {
    return call_delta(params) - 1.0; // Φ(d1) - 1
}
double put_gamma(BS_parametres params) {
    return call_gamma(params);
}
double put_vega(BS_parametres params) {
    return call_vega(params);
}
double put_theta(BS_parametres params) {
    auto [d1, d2] = d1d2(params);
    double term1 = - (params.S0 * norm_pdf(d1) * params.sigma) / (2.0 * std::sqrt(params.T));
    double term2 = + params.r * params.K * std::exp(-params.r * params.T) * norm_cdf(-d2);
    return term1 + term2; // par an
}
double put_rho(BS_parametres params) {
    auto [_, d2] = d1d2(params);
    return -params.K * params.T * std::exp(-params.r * params.T) * norm_cdf(-d2);
}

double theta_per_day(double theta_annual) {
    return theta_annual / 365.0;
}
double vega_per_1pct(double vega_per_1_0) {
    return vega_per_1_0 / 100.0; // variation par +0.01 de sigma
}


int main() {
    BS_parametres BS_para {252, 120 , 0.05, 0.04, 0.5 }; 
    std::cout << call_delta(BS_para) << std::endl;
    return 0 ; 
}