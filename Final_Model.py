import pandas as pd
import numpy as np
import  matplotlib.pyplot  as  plt 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import shap

# Načítanie predspracovaného datasetu
file_path = 'obesity_dataset.csv'
df = pd.read_csv(file_path)

### Načítanie datasetu po PCA
##file_path = 'obesity_dataset_pca.csv'
##df = pd.read_csv(file_path)

# Oddelenie vstupných atribútov (X) a cieľovej premennej (y)
X = df.drop(columns=['NObeyesdad']).values
y = df['NObeyesdad']

# Normalizácia dát
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X = (X-mean) / std

# One-hot encoding
y = tf.keras.utils.to_categorical(y)

# Rozdelenie datasetu na trénovaciu a testovaciu množinu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Definícia modelu
mlp = Sequential()
# Vstupná a skrytá vrstva
mlp.add(Dense(16, activation='relu', input_dim=X.shape[1]))
# 2. skrytá vrstva
mlp.add(Dense(13, activation='relu'))
# 3. skrytá vrstva
mlp.add(Dense(11, activation='relu'))
# Výstupná vrstva
mlp.add(Dense(7, activation='softmax'))

# Kompilácia modelu
mlp.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001),  metrics=['accuracy'])

# Trénovanie modelu
history = mlp.fit(X_train, y_train, epochs=200, batch_size=128, validation_split=0.1, verbose=2)

# Vyhodnotenie modelu na testovacej množine
test_loss, test_accuracy = mlp.evaluate(X_test, y_test, verbose=0)
print(f"Testovacia presnosť: {test_accuracy:.2f}")

# Grafy
plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.legend(loc='best')

plt.figure()
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.legend(loc='best')
plt.show()

# Vytvorenie explainer objektu pre neurónovú sieť - > toto podaril ChatGPT
explainer = shap.Explainer(mlp, X_train)
shap_values = explainer(X_test)

# Vizualizácia priemerného vplyvu atribútov
shap.summary_plot(shap_values, X_test, feature_names=df.columns[:-1])
