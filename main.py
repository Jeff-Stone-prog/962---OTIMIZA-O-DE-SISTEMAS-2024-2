import yfinance as yf
import numpy as np
import pandas as pd

# Lista de ativos que você deseja incluir na carteira
tickers = ['BSBR', 'BBD', 'POSI3.SA', 'BBAS3.SA', 'PBR']

# Coletar dados de preço ajustado
data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']

# Calcular retornos diários
returns = data.pct_change().dropna()

# Calculando o retorno esperado (média) e volatilidade (desvio padrão)
expected_returns = returns.mean()
volatility = returns.std()

print("Retornos esperados:")
print(expected_returns)
print("Volatilidade dos ativos:")
print(volatility)

from scipy.optimize import minimize

# Função para calcular o retorno esperado da carteira
def portfolio_return(weights, expected_returns):
    return np.sum(weights * expected_returns)

# Função para calcular o risco da carteira (usaremos a variância aqui, mas pode-se usar VaR ou CVaR)
def portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Função objetivo (minimização negativa para maximizar o retorno)
def objective_function(weights, expected_returns, cov_matrix, risk_tolerance):
    risk = portfolio_risk(weights, cov_matrix)
    return -portfolio_return(weights, expected_returns) + 1e6 * max(0, risk - risk_tolerance)

# Definindo as restrições
def constraint_sum_of_weights(weights):
    return np.sum(weights) - 1

# Definindo os limites (cada peso deve estar entre 0 e 1)
bounds = [(0, 1) for _ in range(len(tickers))]

# Usando a matriz de covariância para medir o risco
cov_matrix = returns.cov()

# Definir risco máximo aceitável (capacidade da mochila)
risk_tolerance = 1  # Defina o limite de risco aqui

# Chute inicial para os pesos (alocação inicial igual)
initial_weights = np.ones(len(tickers)) / len(tickers)

# Otimização
result = minimize(objective_function, initial_weights, args=(expected_returns, cov_matrix, risk_tolerance), 
                  method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': constraint_sum_of_weights})

# Resultados
optimized_weights = result.x
print("Pesos otimizados:", optimized_weights)

import matplotlib.pyplot as plt
import seaborn as sns

# Fronteira eficiente (retorno x risco)
risks = []
returns = []
for i in range(1000):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    risks.append(portfolio_risk(weights, cov_matrix))
    returns.append(portfolio_return(weights, expected_returns))

plt.figure(figsize=(10, 6))
plt.scatter(risks, returns, c=np.array(returns)/np.array(risks), marker='o')
plt.title('Fronteira Eficiente')
plt.xlabel('Risco (Volatilidade)')
plt.ylabel('Retorno Esperado')
plt.colorbar(label='Índice Sharpe')
plt.show()

# Gráfico de alocação de ativos (pizza)
plt.figure(figsize=(8, 8))
plt.pie(optimized_weights, labels=tickers, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Alocação de Ativos')
plt.show()