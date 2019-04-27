import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

data = pd.read_csv('Admission1.csv')
X = data.drop('Chance of Admit ',axis=1).values

# Создание модели алгоритма PCA
pca = PCA(n_components=3)
# Проецирование обучающего набора до 3ех измерений
X3D = pca.fit_transform(X)
# Вывода коэфицента объясненной дисперсии для каждого компонента из d
coef = pca.explained_variance_ratio_
# Обратная распаковка данных до n измерений
X_recovered = pca.inverse_transform(X3D)
print(X_recovered)
b = 0
for c in coef:
	b += c

print(b,coef)

'''
# Центрирование данных
X_centered = X - X.mean(axis=0)
# Сингулярное разложение матрицы обучающего набора, при этом в матрице
# V содержатся n-галвных компонентов (c1,c2...Cn)
U,s,Vt = np.linalg.svd(X)
# Получение первого главного компонента
c1 = Vt.T[:,0]
# Получение второго главного компонента
c2 = Vt.T[:,1]
# Матрица W3 содержит первые 3 главных компонента (первые 3 столбца V)
W3 = Vt.T[:,:3]
# Проецировние есть суть умножение матрицы X на Wd
X3D = X_centered.dot(W3)

# Чтобы выбрать правильное количество измерений вручную, нужно 
# суммировать вектор коэффицентов объясненной дисперсии до достижения нужной 
# нужной величины, а затем вернуть индекс эл-та+1, это и будет d - 
# количество необходимых измерений. Чтобы реализовать это через sklearn, нужно
# установить значение n_components между 0.0 и 1.0
'''


