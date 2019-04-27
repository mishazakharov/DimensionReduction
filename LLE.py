import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding
# Локальное линейное вложение есть прием нелинейного понижения размерности!
data = pd.read_csv('Admission1.csv')
X = data.drop('Chance of Admit ',axis=1)

# Создание модели Локального линейного вложения
lle = LocallyLinearEmbedding(n_components=3,n_neighbors=10)

# Уменьшение размерности обучающего набора до 3
X_reduced = lle.fit_transform(X)
# Математика описана в книге в двух шагах.