#pragma once

namespace pricers {

struct MCParams {
  double S0, K, r, q, sigma, T;
  int    Npaths{100000};
  bool   antithetic{true};
};

double mc_call_price(const MCParams& p, double* stderr_out = nullptr);
double mc_put_price (const MCParams& p, double* stderr_out = nullptr);

} // namespace pricers