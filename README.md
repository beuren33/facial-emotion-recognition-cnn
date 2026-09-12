# Reconhecedor de Emoções via Webcam

Este projeto faz o reconhecimento de emoções faciais em tempo real usando a webcam. A ideia é simples de enunciar e interessante de resolver: a câmera captura o rosto da pessoa, o sistema localiza onde está a face dentro da imagem e uma rede neural convolucional decide qual das sete emoções aquele rosto está expressando. O resultado aparece na tela, com um retângulo em volta do rosto e o nome da emoção logo acima, tudo acontecendo quadro a quadro enquanto a pessoa se mexe na frente da câmera.

O reconhecimento foi dividido em duas etapas que trabalham juntas. Primeiro entra a detecção de face, feita com o Haar Cascade do OpenCV, que é responsável por dizer em que região da imagem existe um rosto. Só depois disso a emoção é classificada, pois não faria sentido pedir para a rede adivinhar o sentimento olhando a imagem inteira, com parede, cabelo e fundo no meio. Recortando apenas o rosto, a rede recebe exatamente aquilo que interessa, o que deixa a predição bem mais estável.

## Como funciona por dentro

O fluxo em tempo real segue sempre a mesma sequência. Cada quadro da webcam é convertido para escala de cinza, pois a rede foi treinada em imagens de um canal só, e nesse quadro o Haar Cascade procura as faces presentes. Para cada face encontrada, a região do rosto é recortada, redimensionada para 64x64 pixels e normalizada para a faixa de 0 a 1, que é o mesmo pré-processamento usado no treino. Essa padronização é importante: se a imagem chegasse na rede em uma escala diferente da que ela viu durante o aprendizado, a predição sairia furada mesmo com o modelo bem treinado. Por fim, a rede devolve a probabilidade de cada emoção e a de maior valor é escrita na tela.

As sete classes reconhecidas são Surpresa, Medo, Nojo, Feliz, Triste, Raiva e Neutro, sempre nessa ordem, que é a mesma ordem em que os rótulos aparecem no dataset.

## O modelo

O classificador é uma rede neural convolucional montada com Keras, no estilo Sequential, com aproximadamente 950 mil parâmetros. A arquitetura empilha blocos de convolução que vão aumentando a quantidade de feature maps, começando em 32 e dobrando em seguida, cada bloco acompanhado de BatchNormalization para estabilizar o treino, MaxPooling para reduzir a dimensão pela metade e Dropout para desligar parte dos neurônios e conter o overfitting. Foi usada ainda regularização L2 nas convoluções, que penaliza pesos muito grandes e ajuda a rede a não decorar o conjunto de treino.

Um ponto que merece destaque é o tratamento do desbalanceamento do dataset. Ao olhar quantas imagens existiam por classe, ficou claro que a base era bem torta: a emoção mais comum tinha perto de 4.700 imagens, enquanto a mais rara não passava de 281. Nesse cenário, a rede tende a acertar muito as classes fáceis e simplesmente ignorar as raras, porque errar o que é raro quase não pesa no resultado geral. Para corrigir isso foi aplicado o class_weight, que dá mais importância ao erro nas classes menos vistas. Como os pesos calculados ficaram muito extremos numa primeira tentativa e atrapalharam o treino, foi aplicada a raiz quadrada sobre eles, o que suaviza a diferença sem perder o efeito de equilibrar as classes.

O treino também contou com data augmentation, gerando variações das imagens com espelhamento horizontal e pequenas rotações, uma forma de compensar o tamanho modesto da base. Além disso foram usados três callbacks: o ReduceLROnPlateau, que diminui a taxa de aprendizado quando a rede empaca, o EarlyStopping, que interrompe o treino quando a validação para de melhorar, e o ModelCheckpoint, que salva sempre a melhor versão do modelo ao longo das épocas.

## Dados

O conjunto usado no treino tem cerca de 12,3 mil imagens de rostos distribuídas entre as sete emoções, com outras 3,08 mil imagens separadas para teste. As imagens originais têm 100x100 pixels em RGB, mas são reduzidas para 64x64 em escala de cinza antes de entrar na rede, uma escolha que diminui a complexidade e acelera o treino sem prejudicar muito o reconhecimento.

## Resultados

Depois do treino, o modelo foi avaliado no conjunto de teste, com imagens que ele nunca tinha visto, e chegou a uma acurácia em torno de 69%, com perda próxima de 0,89. Durante o treino a acurácia de validação chegou a passar de 71%. Para um problema de sete classes com uma base desbalanceada e relativamente pequena, é um resultado honesto, e a diferença mais visível aparece justamente nas emoções mais raras, que continuam sendo as mais difíceis de acertar mesmo com o class_weight ajudando.

| Métrica | Valor |
| --- | --- |
| Classes | 7 (Surpresa, Medo, Nojo, Feliz, Triste, Raiva, Neutro) |
| Imagens de treino | aproximadamente 12,3 mil |
| Imagens de teste | 3.083 |
| Entrada da rede | 64x64, escala de cinza |
| Parâmetros | cerca de 950 mil |
| Acurácia no teste | por volta de 69% |
| Perda no teste | aproximadamente 0,89 |

## Como rodar

Primeiro instale as dependências:

```bash
pip install -r requirements.txt
```

Para ver o reconhecimento de emoções funcionando ao vivo, com a webcam:

```bash
python webcam_predict.py
```

Se quiser testar apenas a parte de detecção de rosto, sem classificar emoção, existe um script separado só para isso:

```bash
python test_face.py
```

Em ambos os casos a janela do OpenCV abre a webcam e a tecla `q` encerra a execução.

## Estrutura do projeto

```
reconhecedor-emocoes-webcam/
├── webcam_predict.py   # reconhecimento de emocao em tempo real
├── test_face.py        # apenas a deteccao de face com Haar Cascade
├── requirements.txt
├── modelo/
│   ├── modelo_01.h5                          # modelo CNN ja treinado
│   └── haarcascade_frontalface_default.xml   # detector de faces do OpenCV
└── treino/
    └── treino_modelo.ipynb   # notebook com todo o processo de treino
```

## Observações e próximos passos

O gargalo do projeto está no dataset, tanto no tamanho quanto no desbalanceamento, e é natural que a evolução venha por aí. Uma base maior e mais equilibrada, seja por coleta seja por técnicas mais fortes de aumento de dados, provavelmente é o que mais elevaria a acurácia. Também vale experimentar arquiteturas mais profundas ou o uso de transfer learning, partindo de uma rede já treinada em muitas imagens de rosto, algo que costuma render bons ganhos justamente quando os dados próprios são poucos.
