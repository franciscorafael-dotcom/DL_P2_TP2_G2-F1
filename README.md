# DL_P2_TP2_G2-F1

Sistema de classificação de doenças em folhas de videira através de redes neuronais convolucionais (CNN).


---

## Índice

- [Objetivos](#objetivos)
- [Classes](#classes)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Dados e preparação](#dados-e-preparação)
- [Protocolo experimental](#protocolo-experimental)
- [Arquiteturas e treino](#arquiteturas-e-treino)
- [Resultados e seleção do modelo](#resultados-e-seleção-do-modelo)
- [Inferência](#inferência)
- [Instalação e execução](#instalação-e-execução)
- [Limitações e trabalho futuro](#limitações-e-trabalho-futuro)

---

## Objetivos

O projeto procura:

1. Preparar e organizar um conjunto de imagens de folhas de videira.
2. Treinar e comparar diferentes arquiteturas CNN pré-treinadas no ImageNet.
3. Avaliar o desempenho de cada arquitetura em diferentes seeds.
4. Selecionar um modelo com base nos resultados de validação.
5. Disponibilizar uma interface de inferência reutilizável por uma futura aplicação.

O foco atual está na classificação de imagens entre as quatro classes conhecidas pelo dataset. O modelo não é, nesta fase, um sistema geral de diagnóstico de qualquer doença da vinha.

## Classes

O modelo foi treinado para distinguir quatro classes:

| Classe | Identificador |
|---|---:|
| `Black_rot` | 0 |
| `Esca` | 1 |
| `Healthy` | 2 |
| `Leaf_blight` | 3 |

A ordem é importante: corresponde à ordem definida em `data/splits.json` e à ordem dos neurónios da camada de saída. O módulo de inferência lê esta lista do ficheiro de splits, em vez de manter uma segunda lista independente.

## Estrutura do projeto

Estrutura relevante para o pipeline de dados, treino e inferência:

```text
DL_P2_TP2_G2-F1/
│
├── data/
│   ├── fieldvitis/
│   │   ├── Black_rot/
│   │   ├── Esca/
│   │   ├── Healthy/
│   │   └── Leaf_blight/
│   │
│   ├── qualidade/
│   │   ├── manifest_raw.csv
│   │   └── removidas.csv
│   │
│   ├── manifest_clean.csv
│   ├── holdout.csv
│   └── splits.json
│
├── runs/
│   └── P2_runs/
│       ├── resnet50_seed0.keras
│       ├── resnet50_seed1.keras
│       ├── resnet50_seed2.keras
│       ├── ...
│       ├── preds_resnet50_seed1_val.csv
│       └── preds_resnet50_seed1_test.csv
│
├── inference/
│   ├── __init__.py
│   ├── predict.py
│   ├── testar_predict.py
│   └── [imagem de teste, opcional]
│
├── notebooks/
│   ├── preparacao_dados_fieldvitis.ipynb
│   └── treino_cnn_4modelos.ipynb
│
├── AI_LOG.md
├── requirements.txt
└── README.md
```

Os ficheiros `.keras` e `preds_*.csv` são produzidos pelo notebook de treino. A pasta `runs/` contém resultados gerados e não é necessária para guardar o código-fonte. Confirma os nomes e a presença dos ficheiros na tua cópia local do repositório; a estrutura acima representa os componentes usados/documentados neste projeto.

### Ficheiros de dados

- `manifest_raw.csv`: manifesto inicial das imagens.
- `removidas.csv`: registo de imagens removidas durante a preparação. Na execução documentada, o ficheiro tinha zero registos.
- `manifest_clean.csv`: manifesto das imagens consideradas válidas após a preparação.
- `holdout.csv`: lista de caminhos reservados para a avaliação final.
- `splits.json`: classes, caminhos do Holdout e divisões Train/Validation/Test para cada seed.

As imagens são organizadas por classe dentro de `data/fieldvitis/`.

## Dados e preparação

Na execução documentada no notebook:

| Elemento | Quantidade |
|---|---:|
| Imagens no manifesto bruto | 555 |
| Imagens removidas | 0 |
| Imagens no manifesto limpo | 555 |
| Imagens reservadas para Holdout | 50 |
| Imagens disponíveis para desenvolvimento | 505 |

Distribuição das 505 imagens usadas no desenvolvimento, na ordem das classes acima:

| Classe | Imagens |
|---|---:|
| `Black_rot` | 35 |
| `Esca` | 182 |
| `Healthy` | 144 |
| `Leaf_blight` | 144 |

Cada seed utiliza uma divisão com:

| Subconjunto | Imagens |
|---|---:|
| Treino | 352 |
| Validação | 76 |
| Teste interno | 77 |
| **Total** | **505** |

As três seeds usam divisões próprias, definidas em `splits.json`. Em cada seed, Train, Validation e Test são disjuntos e, em conjunto, cobrem as 505 imagens não-Holdout.

### Proteção do Holdout

O Holdout contém 50 imagens e é mantido separado do processo de desenvolvimento. O notebook verifica que:

- os caminhos de `holdout.csv` coincidem com os caminhos de Holdout em `splits.json`;
- não existem interseções entre Holdout e os conjuntos Train, Validation e Test;
- os conjuntos de cada seed cobrem exatamente as imagens não-Holdout;
- os ficheiros referenciados existem no diretório do dataset.

Na execução registada, o notebook confirmou: **`Imagens Holdout carregadas: 0`**.

O Holdout deve ser usado apenas na avaliação final, depois de concluída a seleção do modelo. Os seus resultados não devem ser usados para escolher arquiteturas, ajustar hiperparâmetros ou voltar a treinar o modelo.

## Protocolo experimental

Foram comparadas quatro arquiteturas pré-treinadas no ImageNet:

- ResNet50
- EfficientNet-B0
- ConvNeXt-Tiny
- DenseNet121

Cada arquitetura foi treinada com três seeds: `0`, `1` e `2`.

O tamanho de entrada usado é `224 × 224` píxeis, com imagens RGB. O batch size é 32.

### Pré-processamento das imagens

Durante o carregamento:

1. A orientação EXIF é corrigida.
2. A imagem é convertida para RGB.
3. A imagem é redimensionada para `224 × 224`.
4. Os valores são convertidos para `float32`.

Depois, é aplicado o pré-processamento específico da arquitetura:

| Arquitetura | Pré-processamento |
|---|---|
| ResNet50 | `keras.applications.resnet50.preprocess_input` |
| EfficientNet-B0 | `keras.applications.efficientnet.preprocess_input` |
| ConvNeXt-Tiny | `keras.applications.convnext.preprocess_input` |
| DenseNet121 | `keras.applications.densenet.preprocess_input` |

**Importante:** na inferência é obrigatório manter o pré-processamento correspondente ao modelo. `inference/predict.py` aplica o pré-processamento da ResNet50.

### Aumento de dados (data augmentation)

Durante o treino é usada uma sequência de aumento de dados:

- `RandomFlip("horizontal_and_vertical")`
- `RandomRotation(0.15)`
- `RandomZoom(0.2)`

É também aplicada uma camada `Dropout(0.3)` antes da camada de classificação.

### Estratégia de transferência de aprendizagem

O treino decorre em duas fases:

**Fase 1 — Treino da cabeça de classificação**

- A base pré-treinada no ImageNet fica congelada.
- Apenas a camada final de classificação é treinada.
- Épocas máximas: 5.
- Otimizador: Adam.
- Learning rate: `1e-3`.

**Fase 2 — Fine-tuning**

- A base convolucional é descongelada.
- As camadas `BatchNormalization` permanecem congeladas.
- Épocas máximas: 25.
- Otimizador: Adam.
- Learning rate: `1e-5`.
- Early stopping com base no Macro-F1 de validação.
- Patience: 6 épocas.

A função de perda é `sparse_categorical_crossentropy`. A camada de saída tem quatro unidades e ativação `softmax`.

São usados pesos por classe (`class_weight`) calculados a partir da distribuição das classes no conjunto de treino, para compensar o desequilíbrio entre classes.

O callback personalizado `EarlyStoppingMacroF1` calcula o Macro-F1 no conjunto de validação no final de cada época, guarda os melhores pesos e restaura-os no fim do treino.

## Resultados e seleção do modelo

### Macro-F1 máximo de validação por seed

O critério principal de seleção é o melhor Macro-F1 obtido no conjunto de validação durante o fine-tuning.

| Arquitetura | Seed 0 | Seed 1 | Seed 2 | Média |
|---|---:|---:|---:|---:|
| ResNet50 | 0.9563 | **0.9667** | 0.9501 | **0.9577** |
| DenseNet121 | 0.9160 | 0.9467 | 0.9608 | 0.9412 |
| EfficientNet-B0 | 0.9608 | 0.9182 | 0.8772 | 0.9187 |
| ConvNeXt-Tiny | 0.9172 | 0.8932 | 0.8809 | 0.8971 |

A ResNet50 apresenta o melhor Macro-F1 médio de validação entre as quatro arquiteturas. A execução `ResNet50 / seed 1` obteve o melhor resultado individual de validação: **0.9667**.

Por esse motivo, o modelo atualmente selecionado para inferência é:

```text
runs/P2_runs/resnet50_seed1.keras
```

A seleção da arquitetura e da seed é baseada nos resultados de validação. Os resultados do conjunto de teste interno abaixo são apresentados como análise adicional; não substituem a avaliação final no Holdout.

### Resultados médios no teste interno

O notebook também calcula Accuracy, Macro-F1 e Macro-Recall no conjunto de teste interno, e apresenta a média das três seeds por arquitetura.

| Arquitetura | Accuracy | Macro-F1 | Macro-Recall |
|---|---:|---:|---:|
| ResNet50 | **0.9784** | **0.9669** | **0.9710** |
| DenseNet121 | 0.9524 | 0.9301 | 0.9611 |
| ConvNeXt-Tiny | 0.9134 | 0.8753 | 0.9148 |
| EfficientNet-B0 | 0.9004 | 0.8639 | 0.9070 |

Estes valores correspondem ao teste interno das divisões definidas em `splits.json`, não ao Holdout.

### Custo computacional

O notebook inclui ainda uma medição de número de parâmetros e latência, usando os modelos da seed 0 e imagens de validação. Os valores registados foram:

| Arquitetura | Parâmetros (milhões) | ms/imagem em lote | ms/imagem isolada |
|---|---:|---:|---:|
| ResNet50 | 23.60 | 14.7 | 96.1 |
| EfficientNet-B0 | 4.05 | 14.8 | 115.9 |
| ConvNeXt-Tiny | 27.82 | 32.2 | 1072.8 |
| DenseNet121 | 7.04 | 22.1 | 196.9 |

Os tempos dependem do hardware, da versão das bibliotecas e das condições de execução; não devem ser interpretados como garantias de latência em produção.

## Inferência

O módulo `inference/predict.py` disponibiliza a função `predict_image()`, que recebe o caminho de uma imagem e devolve:

- `predicted_class`: classe prevista;
- `confidence`: valor softmax da classe prevista;
- `probabilities`: valores softmax para todas as classes.

O modelo é carregado uma única vez e reutilizado nas chamadas seguintes.

### Exemplo de utilização

A partir da raiz do projeto:

```python
from inference.predict import predict_image

result = predict_image("caminho/para/folha.jpg")

print(result)
```

Exemplo de estrutura de resposta (valores ilustrativos):

```python
{
    "predicted_class": "Esca",
    "confidence": 0.94,
    "probabilities": {
        "Black_rot": 0.02,
        "Esca": 0.94,
        "Healthy": 0.01,
        "Leaf_blight": 0.03
    }
}
```

Para testar uma imagem colocada dentro da pasta `inference/`, pode ser usado um script como `inference/testar_predict.py`:

```python
from pathlib import Path
from inference.predict import predict_image

image_path = Path(__file__).parent / "Esca_104.jpg"

result = predict_image(image_path)
print(result)
```

O caminho do modelo configurado no módulo é:

```text
runs/P2_runs/resnet50_seed1.keras
```

O módulo também lê a ordem das classes a partir de `data/splits.json`. Por isso, para executar a inferência, são necessários o ficheiro do modelo e o ficheiro `splits.json`, além das dependências Python.

### Como interpretar a saída

Os valores de `probabilities` são as saídas softmax do classificador. O maior valor determina a classe prevista.

A `confidence` é o valor softmax associado à classe escolhida. **Não deve ser interpretada automaticamente como a probabilidade real de a previsão estar correta**, nem como uma medida de calibração.

O classificador foi treinado para escolher sempre uma das quatro classes conhecidas. Não existe ainda um mecanismo validado para rejeitar fotografias que não sejam folhas de videira, imagens de baixa qualidade ou doenças que não pertençam às classes do treino.

## Instalação e execução

O projeto deve ser executado num ambiente Python com as dependências compatíveis com o notebook e com o modelo Keras guardado.

Dependências principais:

- TensorFlow
- Keras
- NumPy
- Pandas
- Pillow
- scikit-learn
- Jupyter Notebook ou JupyterLab (para executar os notebooks)

Exemplo de instalação das dependências principais num ambiente virtual (recomendado, mas não obrigatório):
#### Linux/macOS
```bash
python -m venv .venv
source .venv/bin/activate
```
#### Windows (CMD)
```bash
python -m venv .venv
.venv\Scripts\activate.bat
```
#### Windows (PowerShell)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```
### Instalar as depedências:
```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A instalação de TensorFlow com suporte GPU depende da versão de Python, do sistema operativo e das versões compatíveis de CUDA/cuDNN. Consulta a documentação oficial do TensorFlow para configurar a GPU no ambiente utilizado.

Para testar a inferência:

```bash
python inference/testar_predict.py
```

Para executar os notebooks, inicia o Jupyter na raiz do projeto:

```bash
jupyter lab
```

> A lista acima é uma lista de dependências principais, não um ficheiro de versões fixadas. Para reproduzir exatamente os resultados, recomenda-se criar e manter um `requirements.txt` ou ficheiro de ambiente com as versões efetivamente usadas no treino.


'''


