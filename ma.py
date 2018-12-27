import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statistics as stat


mu = 1.5
fig = plt.gcf()
fig.set_size_inches(14, 10)
N = 100
rnd = np.random.exponential(mu, N)

# Гистограмма

plt.subplot(3, 3, 1)
nbins =10
plt.hist(rnd, nbins, normed=True, facecolor='black', alpha=0.6, label=str(N))
plt.legend(loc='upper right')
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$H$', fontsize=14)
plt.axis([0, mu+9, 0, 0.7])
plt.grid(True)



# Функция плотности вероятности


xmean=stat.mean(rnd)
x=np.linspace(0, xmean+200, 10000)
f=(xmean**-1)*(np.e**(-(xmean**-1)*x))
plt.plot(x, f)


# Функция эмпирического кумулятивного распределения


plt.subplot(3, 3, 2)
ecdf = sm.distributions.ECDF(rnd)
x = np.linspace(min(rnd), max(rnd))
F = ecdf(x)
plt.step(x, F, label=str(N))
plt.legend(loc='upper left' )
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$P$', fontsize=14)
plt.axis([-0.5, mu+9, 0, 1.4])
plt.grid(True)




#Теоретическая кумулята


x = np.linspace(0, xmean+200, 10000)
f = 1-np.e**(-(xmean**-1)*x)
plt.plot(x, f)
plt.show()




print()
print('Выборочное среднее:')
print(sum(rnd) / len(rnd))
print()
import statistics
print('Выборочная дисперсия:')
print(statistics.variance(rnd))
print()
print('Выборочное стандартное отклонение:')
import statistics
print(statistics.stdev(rnd))



