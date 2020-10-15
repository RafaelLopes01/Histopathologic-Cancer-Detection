# Redes Neurais - Detecção de Cancêr em Imagens

Projeto desenvolvido a partir de uma base de dados do [Kaggle](https://www.kaggle.com/), que se chama [histopathologic-cancer-detection](https://www.kaggle.com/c/histopathologic-cancer-detection), que é composta por 200 mil imagens para treinamento, 60 mil para teste e o objetivo é identificar a existência de cancêr nas imagens de microscópio.

Esse projeto eu desenvolvijunto com meu grupo para ser entregue no trabalho de conclusão do módulo de deep learning da minha pós graduação em Data Science.

### Projeto
Desenvolvemos uma rede neural convolucional (CNN) para classificação das imagens, utilizando Python, Tensorflow e o Keras.

A estrutura da rede conta com três camadas de convolução, onde cada camada tem a seguinte configuração: inicia com uma camada de convolução que gera 128 mapas de características da imagem original(96x96) com a função de ativação 'relu'. Em seguida uma camada para normalização dos dados usando o BatchNormalization e outra camada de MaxPooling que é usada para reduzir a dimensionalidade da imagem e pegar suas principais características.

No final das três camadas convolucionais usamos uma camada de Flatten para transformar a matriz de saída do ultimo MaxPooling em um vetor, que será usado como entrada dos neurônios das camadas densas.



### Resultado


