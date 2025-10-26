# Option Pricing & Hedging Engine (C++/Python) 📈

A high-performance quantitative finance library for option pricing, risk analysis (Greeks), and backtesting of a delta-hedging strategy. The core engine is built in C++ for speed, with a Python interface for flexible analysis and deployment.

---

## ✨ Key Features

* **Multi-Model Pricing:** Implements Black-Scholes, Binomial Trees (CRR), and Monte Carlo models for European call options.
* **Comprehensive Risk Analysis:** Calculates and validates key Greeks (Delta, Gamma, Vega, Theta) to manage portfolio risk.
* **Rigorous Model Validation:** Provides in-depth analysis of model performance, including error distribution by maturity and sanity checks on Greeks.
* **Delta-Hedging Backtesting Engine:** Simulates a delta-neutral strategy, tracking daily P&L, NAV, transaction costs, and key performance metrics.
* **Interactive Dashboard:** A Streamlit web application to visualize model performance, portfolio evolution, and hedging results.
* **Automated Reporting:** Generates analysis reports via the OpenAI (GPT-4) API.

---

## 🛠️ Technical Stack

* **High-Performance Computing:** C++17
* **Scripting & Analysis:** Python 3.12, Pybind11
* **Data Management:** Supabase (PostgreSQL)
* **Data Science:** Pandas, NumPy
* **Web Dashboard:** Streamlit
* **Deployment:** Docker

---

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
    docker run option-pricer
    ```

---

## 🖥️ Interactive Dashboard

To launch the Streamlit web application and explore the results interactively, run the following command. This will make the dashboard available in your browser.

```bash
docker run -p 8501:8501 option-pricer streamlit run python/dashboard.py
```
Then, open your web browser and go to **http://localhost:8501**.

---

## ⚖️ License

This project is distributed under the MIT License.
