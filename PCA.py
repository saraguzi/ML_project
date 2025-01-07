import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Načítanie datasetu
file_path = 'obesity_dataset.csv'
df = pd.read_csv(file_path)

# Oddelenie vstupných atribútov (X) a cieľovej premennej (y)
X = df.drop(columns=['NObeyesdad'])
y = df['NObeyesdad']

# Normalizácia dát
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X = (X - mean) / std

# Zoznam pre ukladanie rozpylu pre rôzny počet komponentov
variances = []

# Postupné vykonávanie PCA pre rôzny počet komponentov
for i in range(1, 17):
    pca = PCA(n_components=i)
    pca.fit(X)
    variances.append(pca.explained_variance_ratio_.sum())
    print(f"Pre počet komponentov {i}: rozptyl {variances[i - 1]}.")

# Vizualizácia - ako sa mení rozptyl pre rôzny počet komponentov
plt.figure(figsize=(8, 5))
plt.plot(range(1, 17), variances, marker='o', linestyle='-')
plt.xticks(np.arange(1, 17, 1))
plt.yticks(np.arange(0, 1.05, 0.05))
plt.title('Ako sa mení rozptyl pre rôzny počet komponentov')
plt.xlabel('Počet komponentov')
plt.ylabel('Rozptyl')
plt.grid(True)
plt.show()

# Chceme počet komponentov, ktorý zachová aspoň 95% rozptylu
optimal_n_components = next(i for i, variance in enumerate(variances, start=1) if variance >= 0.95)

# Aplikácia PCA s optimálnym počtom komponentov
pca_optimal = PCA(n_components=optimal_n_components)
X_pca_optimal = pca_optimal.fit_transform(X)

# Uloženie transformovaného datasetu
df_pca = pd.DataFrame(X_pca_optimal, columns=[f'PC{i+1}' for i in range(optimal_n_components)])
df_pca['NObeyesdad'] = y
output_file_path = 'obesity_dataset_pca.csv'
df_pca.to_csv(output_file_path, index=False)
