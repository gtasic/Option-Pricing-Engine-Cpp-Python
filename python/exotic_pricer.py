import numpy as np
import sys
sys.path.append("/workspaces/finance-/build")
import finance as fn


class ExoticPricer:
    
    def barrier_option(self, S0, K, B, T, r, sigma, barrier_type='down-and-out'):

        lambda_param = (r + 0.5 * sigma**2) / sigma**2
        y = np.log(B**2 / (S0 * K)) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
        x1 = np.log(S0 / B) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
        y1 = np.log(B / S0) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
        

        BS_para = fn.BS_parametres(S0, K, T, r, sigma)
        vanilla_price = fn.call_price(BS_para)
        BS_param = fn.BS_parametres(B**2/S0, K, T, r, sigma)
        barrier_adjustment = (B/S0)**(2*lambda_param) * fn.call_price(BS_param)
        
        return vanilla_price - barrier_adjustment if S0 > B else 0
        
    def asian_option(self, S0, K, T, r, sigma, n_observations=12):

        dt = T / n_observations
        paths = 10000
        payoffs = []
        
        for _ in range(paths):
            S_path = [S0]
            for t in range(n_observations):
                dW = np.random.randn() * np.sqrt(dt)
                S_next = S_path[-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*dW)
                S_path.append(S_next)
            
            avg_S = np.mean(S_path)
            payoffs.append(max(avg_S - K, 0))
        
        return np.mean(payoffs) * np.exp(-r*T)
    
    def digital_option(self, S0, K, T, r, sigma):
        d2 = (np.log(S0/K) + (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return np.exp(-r*T) * fn.norm_cdf(d2)
    
    def digital_option_put(self, S0, K, T, r, sigma):
        d2 = (np.log(S0/K) + (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return np.exp(-r*T) * fn.norm_cdf(-d2)

def test_exotics():
    pricer = ExoticPricer()
    BS_para = fn.BS_parametres(100, 100, 1, 0.05, 0.2)
    vanilla = fn.call_price(BS_para)
    barrier = pricer.barrier_option(100, 100, 80, 1, 0.05, 0.2)
    assert barrier < vanilla, "Arbitrage! Barrier devrait être moins cher"
    
    digital_call = pricer.digital_option(100, 100, 1, 0.05, 0.2)
    digital_put = pricer.digital_option_put(100, 100, 1, 0.05, 0.2)
    zcb = np.exp(-0.05 * 1)
    assert abs((digital_call + digital_put) - zcb) < 0.01, "Put-Call Parity violated"
    
    print("✅ All tests passed")

test_exotics()
