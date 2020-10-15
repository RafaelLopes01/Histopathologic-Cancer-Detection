# Redes Neurais - Detecção de Câncer em Imagens

Projeto desenvolvido a partir de uma base de dados do [Kaggle](https://www.kaggle.com/), que se chama [histopathologic-cancer-detection](https://www.kaggle.com/c/histopathologic-cancer-detection), que é composta por 200 mil imagens para treinamento, 60 mil para teste e o objetivo é identificar a existência de câncer nas imagens de microscópio.

Esse projeto eu desenvolvi junto com meu grupo para ser entregue no trabalho de conclusão do módulo de __deep__ __learning__ da minha pós graduação em Data Science.

### Projeto
Desenvolvemos uma rede neural convolucional __(CNN)__ para classificação das imagens, utilizando __Python__, __Tensorflow__ e o __Keras__.

A estrutura da rede conta com três camadas de convolução, onde cada camada tem a seguinte configuração: inicia com uma camada de convolução que gera 128 mapas de características da imagem original(96x96) com a função de ativação __'relu'__. Em seguida uma camada para normalização dos dados usando o __BatchNormalization__ e outra camada de __MaxPooling__ que é usada para reduzir a dimensionalidade da imagem e pegar suas principais características.

No final das três camadas convolucionais usamos uma camada de __Flatten__ para transformar a matriz de saída do ultimo __MaxPooling__ em um vetor, que será usado como entrada para os neurônios das camadas densas.

Usando a saída do vetor criamos três camadas densas, cada uma com 128 neurônios. Cada camada é seguida por outra de __Dropout__, onde está configurado para inativar 20% dos neurônios, afim de evitar o __Overfitting__.

No final das camadas densas, inserimos uma camada de saída de um unico neurônio usando a função de ativação __'sigmoid'__ que retorna um valor entre 0 e 1, onde o valor mais próximo de 1 representa o diagnóstico positivo de câncer.

Após montar a estrutura da rede compilamos ela usando o __adam__ como método de descida do gradiente, o __binary_crossentropy__ como função de perda e a __accuracy__ como métrica.

Para realizar o treinamento da rede neural, dividimos a nossa base de treinamento onde tinhamos todos os resultados da classificação das imagens em 75% para o treinamento e 25% para a validação da rede. 

O treinamento foi realizado por 100 épocas e com 100 etapas de validação por época, onde durou aproximadamente 20 horas usando uma GPU.

### Resultado

Obtivemos uma média de acurácia em torno de 94% durante o treinamento e 91% na validação. Ao fazer as predições na base de teste do Kaggle, obtivemos 91% de acurácia em uma base que não sabiamos a classificação das imagens.

