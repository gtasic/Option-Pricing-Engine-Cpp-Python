#pragma once
#include <utility>

namespace pricers {

struct BSParams {
  double S0, K, r, q, sigma, T;
};

double black_scholes_call(const BSParams& p);
double black_scholes_put (const BSParams& p);

// Greeks (retour simple ou struct dédié)
struct Greeks { double delta, gamma, vega, theta, rho; };
Greeks bs_call_greeks(const BSParams& p);
Greeks bs_put_greeks (const BSParams& p);

} 