# A1 - Mineração de Dados: Video Game Sales

Repositório contendo o projeto de Mineração de Dados desenvolvido para a disciplina _Mineração de Dados_, cujo objetivo é analisar e prever a preferência regional de vendas de jogos de videogame. O trabalho foi dividido em duas partes:

1. **Implementação e envio do código** (valor 5,0)
2. **Apresentação dos integrantes** (valor 5,0)

---

## Descrição do Projeto

O objetivo principal é realizar uma análise de dados de vendas de videogames, a partir de uma base de dados pública (**Video Game Sales**), utilizando técnicas de **Machine Learning**. As etapas principais são:

1. **Coleta** e **exploração** dos dados.
2. **Pré-processamento**, lidando com valores faltantes e transformando colunas categóricas em numéricas.
3. Aplicação de **três algoritmos de classificação**:
   - KNN (K-Nearest Neighbors)
   - MLP (Multi-Layer Perceptron)
   - RandomForest (Random Forest)
4. **Avaliação** dos modelos por meio de três métricas (acurácia, precisão e recall).
5. Comparação dos resultados para determinar qual modelo apresentou melhor desempenho.

---

## Estrutura do Repositório

Dentro do repositório, encontramos a seguinte organização de pastas e arquivos (exemplo):

```
A1-DataMiningVideoGameSales/
├── data/
│   └── Video+Game+Sales/
│       ├── vg_data_dictionary.csv
│       └── vgchartz-2024.csv
├── venv/                (pasta do ambiente virtual - opcional, não versionado)
├── A1_DataMining.py     (script principal de análise e classificação)
├── relatorio_vendas_games.pdf (relatório em PDF com os gráficos gerados)
└── README.md            (este arquivo)
```

- **data/Video+Game+Sales/**: Contém os arquivos CSV originais de vendas de videogame.
- **A1_DataMining.py**: Script Python com o passo a passo de análise, pré-processamento, e aplicação dos modelos de classificação.
- **relatorio_vendas_games.pdf** (opcional): PDF com os gráficos salvos automaticamente (se configurado no script).
- **README.md**: Documento explicativo sobre o projeto (este arquivo).

---

## Como Executar o Projeto

### 1. Clonar o Repositório

No terminal, execute:

```bash
git clone https://github.com/hiag0liveira/A1-DataMiningVideoGameSales.git
cd A1-DataMiningVideoGameSales
```

### 2. Criar e Ativar Ambiente Virtual (Recomendado)

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependências

Com o ambiente virtual ativo, rode:

```bash
pip install -r requirements.txt
```

Caso você não possua um `requirements.txt`, instale manualmente as bibliotecas:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

### 4. Executar o Script

Para rodar a análise e gerar os gráficos:

```bash
python A1_DataMining.py
```

- Foi incluido um código para geração de PDF, ao final da execução será criado um arquivo `relatorio_vendas_games.pdf`.

---

## Principais Bibliotecas Utilizadas

- **pandas**: Manipulação e análise de dados.
- **numpy**: Suporte para operações numéricas e uso de arrays.
- **seaborn** e **matplotlib**: Visualizações de dados e plotagem de gráficos.
- **scikit-learn**: Algoritmos de Machine Learning (classificação, divisão de dados, métricas, etc.).

---

## Resultados

No final da análise, são exibidos:

1. **Métricas** dos modelos (acurácia, precisão e recall).
2. **Matrizes de confusão** para cada algoritmo.
3. **Gráficos** comparativos da acurácia dos modelos.

Concluímos qual algoritmo obteve melhor desempenho na classificação da **região preferencial** de vendas (region_preference) para os jogos de videogame analisados.

---

## Créditos / Integrantes do Grupo

- **Hiago De Oliveira**: responsável pelo ambiente de produção.
- **Lucas Rangel**: responsável pela analise do CSV e leitura.
- **Matheus Rocha**: responsável pela escolha do algoritmo escolhido.

---
