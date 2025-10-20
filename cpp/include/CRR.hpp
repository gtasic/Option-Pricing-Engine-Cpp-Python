#pragma once

namespace pricers {

enum class OptionType { Call, Put };

struct CRRParams {
  double S0, K, r, q, sigma, T;
  int    N;
  OptionType type;
};

double crr_price_european(const CRRParams& p);
double crr_price_american(const CRRParams& p);

} // namespace pricers