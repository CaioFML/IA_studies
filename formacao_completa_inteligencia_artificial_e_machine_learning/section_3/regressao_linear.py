# regressao linear
# Correlacao > inclinacao > interceptacao > previsao

# formulas
# correlation pearson
# r = sum((x - x_mean) * (y - y_mean)) / sqrt(sum((x - x_mean) ** 2) * sum((y - y_mean) ** 2))

# inclination
# b = r * (std(y) / std(x))

# intercept
# a = y_mean - b * x_mean

# prediction
# y = a + b * x

from numpy import *

class LinearRegression:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.__correlation_coefficient = self.__correlation_coefficient()
    self.__inclination = self.__inclination()
    self.__interception = self.__interception()

  def __correlation_coefficient(self):
    covariance = cov(self.x, self.y, bias = True)[0][1]
    x_var = var(self.x)
    y_var = var(self.y)
    return covariance / sqrt(x_var * y_var)

  def __inclination(self):
    stdx = std(self.x)
    stdy = std(self.y)
    return self.__correlation_coefficient * (stdy / stdx)

  def __interception(self):
    meanx = mean(self.x)
    meany = mean(self.y)
    return meany - meanx * self.__inclination

  def predict(self, z):
    return self.__interception + (self.__inclination * z)

x = array([1, 2, 3, 4, 5])
y = array([2, 4, 6, 8, 10])

lr = LinearRegression(x, y).predict(6)
print(lr) # 12.0
