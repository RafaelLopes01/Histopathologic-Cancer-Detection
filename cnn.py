from tensorflow import keras
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

#Rede Covolucional
#Primeira Etapa de Convolução
classificador = keras.models.Sequential()
classificador.add(keras.layers.Conv2D(128, (2, 2), input_shape=(96,96,3), activation='relu'))
classificador.add(keras.layers.BatchNormalization())
classificador.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

#Segunda Etapa de Convolução
classificador.add(keras.layers.Conv2D(128, (2, 2), activation='relu'))
classificador.add(keras.layers.BatchNormalization())
classificador.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

#Terceira Etapa de Convolução
classificador.add(keras.layers.Conv2D(128, (2, 2), activation='relu'))
classificador.add(keras.layers.BatchNormalization())
classificador.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

#Flatten
classificador.add(keras.layers.Flatten())

#Rede Neural Densa com 3 Camadas
classificador.add(keras.layers.Dense(units=128, activation='relu', input_dim = 15488))
classificador.add(keras.layers.Dropout(0.2))
classificador.add(keras.layers.Dense(units=128, activation='relu'))
classificador.add(keras.layers.Dropout(0.2))
classificador.add(keras.layers.Dense(units=128, activation='relu'))
classificador.add(keras.layers.Dropout(0.2))
classificador.add(keras.layers.Dense(units=1, activation='sigmoid'))

#Compile
classificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classificador.summary()

#Data Augmentation
gerador_treinamento = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                   rotation_range=7,
                                                                   horizontal_flip=True,
                                                                   shear_range=0.2,
                                                                   height_shift_range=0.07,
                                                                   zoom_range=0.2)

gerador_teste = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

data = pd.read_csv('train_labels.csv')
data['id'] = data['id'] + '.tif'
data['label'] = data['label'].astype('str')
df_train, df_valid = train_test_split(data, test_size = 0.25, random_state = 0)

base_treinamento = gerador_treinamento.flow_from_dataframe(
    df_train,
    directory='train',
    x_col="id",
    y_col="label",
    weight_col=None,
    target_size=(96, 96),
    color_mode="rgb",
    classes=None,
    class_mode="binary",
    batch_size=32,
    shuffle=True,
    seed=42,
    interpolation="nearest",
    validate_filenames=True
)

base_teste = gerador_teste.flow_from_dataframe(
    df_valid,
    directory='train',
    x_col="id",
    y_col="label",
    weight_col=None,
    target_size=(96, 96),
    color_mode="rgb",
    classes=None,
    class_mode="binary",
    batch_size=32,
    shuffle=True,
    seed=42,
    interpolation="nearest",
    validate_filenames=True
)

#Treinamento
classificador.fit_generator(base_treinamento, epochs=100,
                            validation_data=base_teste, validation_steps= 100)

#Horario do Fim do Treinamento
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))

#Salvando os Pesos
classificador_json = classificador.to_json()
with open('classificador.json', 'w') as json_file:
    json_file.write(classificador_json)
    
classificador.save_weights('classificador_weights.h5')

#Indices
base_treinamento.class_indices

