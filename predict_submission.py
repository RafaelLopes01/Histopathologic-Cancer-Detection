#import
from tensorflow import keras
import pandas as pd

#Importando o Classificador
arquivo = open('classificador.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

#Carregando os Pesos
classificador = keras.models.model_from_json(estrutura_rede)
classificador.load_weights('classificador_weights.h5')

##Data Augmentation Base de Teste
gerador_teste = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
df_valid = pd.read_csv('sample_submission.csv')
df_valid['id'] = df_valid['id'] + '.tif'
df_valid['label'] = df_valid['label'].astype('str')

base_teste = gerador_teste.flow_from_dataframe(
    df_valid,
    directory='test',
    x_col='id',
    y_col='label',
    target_size=(96, 96),
    color_mode="rgb",
    class_mode="binary",
    batch_size=32,
    seed=42,
    shuffle=False,
)

#Predict
results = classificador.predict(base_teste)

#Gerando Submission para o Kaggle      
for index, row in df_valid.iterrows():
    df_valid.at[index, 'label'] = results[index][0]
    
df_valid['id'] = df_valid['id'].str.replace('.tif', '')
df_valid.to_csv('submission.csv', index=False)
