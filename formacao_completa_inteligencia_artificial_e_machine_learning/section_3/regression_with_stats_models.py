import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns

base = pd.read_csv('mt_cars.csv')

print(base.shape)

print(base.head())

base = base.drop(['Unnamed: 0'], axis = 1)

print(base.head())

corr = base.corr()

print(corr)

# sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')

# plt.show()

column_pairs = [('mpg', 'cyl'), ('mpg', 'disp'), ('mpg', 'hp'), ('mpg', 'wt'), ('mpg', 'qsec'), ('mpg', 'vs'), ('mpg', 'am'), ('mpg', 'gear'), ('mpg', 'carb')]

# n_plots = len(column_pairs)
# fig, axes = plt.subplots(nrows=n_plots, ncols=1, figsize=(6,4 * n_plots))

# for i, pair in enumerate(column_pairs):
#   x_col, y_col = pair
#   sns.scatterplot(x=x_col, y=y_col, data=base, ax=axes[i])
#   axes[i].set_title(f'{x_col} vs {y_col}')

# plt.tight_layout()
# plt.show()

# aic 156.6 bic 162.5
# modelo = sm.ols(formula='mpg ~ wt + disp + hp', data=base)

# aic 165.1 bic 169.5
# modelo = sm.ols(formula='mpg ~ disp + cyl', data=base)

#aic 179.1 bic 183.5
modelo = sm.ols(formula='mpg ~ drat + vs', data=base)

modelo = modelo.fit()
print(modelo.summary())

residuos = modelo.resid
print(residuos)
plt.hist(residuos, bins=20)
plt.xlabel("residuos")
plt.ylabel("frequencia")
plt.title("Histograma dos residuos")
plt.show()

stats.probplot(residuos, dist="norm", plot=plt)
plt.title("QQ plot dos residuos")
plt.show()

# h0 - dados estao normalmente distribuidos
# p <= 0.05 rejeito a hipotese nula (nao estao normalmente distribuidos)
# p > 0.05 nao e possivel rejeitar a hipotese nula
stat, pval = stats.shapiro(residuos)
print(f'Shapiro-Wilk statistica: {stat:.3f}, p-value: {pval:.3f}')