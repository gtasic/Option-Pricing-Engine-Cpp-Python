# Option Pricing & Hedging Engine (C++/Python) 📈

A high-performance quantitative finance project for option pricing, risk analysis (Greeks), backtesting of a delta-hedging strategy and volatility arbitrage. The core engine is built in C++ for speed, with a Python interface for flexible analysis and deployment.

---

---

## ✨ Key Features

* **⚡ Multi-Model Pricing:** High-performance C++ implementations of **Black-Scholes**, **Binomial Trees (CRR)**, and **Monte Carlo** simulations.
* **📊 Risk Analysis (Greeks):** Real-time calculation of Delta, Gamma, Vega, and Theta for portfolio risk management.
* **🤖 Machine Learning:** Volatility prediction using **XGBoost** trained on real market data (Yahoo Finance).
* **📉 Delta-Hedging Engine:** Backtesting simulator tracking P&L, NAV, and transaction costs for delta-neutral strategies.
* **📈 Volatility Arbitrage:** Smile construction and arbitrage opportunity detection.
* **🖥️ Interactive Dashboard:** Full-stack web app (Streamlit and Python) deployed via Docker.

---

## 🛠️ Technical Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Core Engine** | **C++20** | Optimized for numerical computation speed. |
| **Interface** | **Pybind11** | Zero-copy binding between C++ backend and Python. |
| **Data & ML** | **Pandas, XGBoost** | Data manipulation and volatility forecasting. |
| **Database** | **Supabase (PostgreSQL)** | Storage for historical data and backtest results. |
| **Frontend** | **Streamlit, Plotly** | Interactive visualization and reporting. |
| **DevOps** | **Docker** | Containerized environment for reproducible builds. |

---

## 🏗️ Project Structure

```text
├── cpp/                  # C++ Core (Pricing engines, MC simulations)
├── python/               # Python Bindings & Data Analysis
│   ├── dashboard         # Streamlit Entry point
│   └── ml_models/        # XGBoost volatility models
├── Dockerfile            # Multi-stage build configuration
├── requirements.txt      # Python dependencies
└── CMakeLists.txt        # Build system config
```


## 🚀 Quick Start with Docker

This project is fully containerized, ensuring a simple and reproducible setup. Make sure Docker is installed on your machine.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/gtasic/Option-Pricing-Engine-Cpp-Python.git](https://github.com/gtasic/Option-Pricing-Engine-Cpp-Python.git)
    cd Option-Pricing-Engine-Cpp-Python
    ```

2.  **Build the Docker image:**
    *(This command assembles the complete environment. It may take a few minutes on the first run.)*
    ```bash
    docker build -t option-pricer .
    ```

3.  **Run the analysis script:**
    *(This command runs the default script to validate the pricing models.)*
    ```bash
    docker run -p 8501:8501 option-pricer
    ```

---

## 🖥️ Interactive Dashboard

To see the intercative dashboard, you can go on the depoyed web-page. 

https://huggingface.co/spaces/gtasic/Volatility-Arbitrage-Engine


---

##📬 Contact
- Developer: Gaspard Tasic
- https://www.linkedin.com/in/gaspard-tasic-3553692a2/

## ⚖️ License

This project is distributed under the MIT License.
