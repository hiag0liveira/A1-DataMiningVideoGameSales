import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Extras para PDF
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rcParams

# Para capturar o texto do console
import io
import sys
from contextlib import redirect_stdout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

# Ajusta fonte menor para caber mais texto nas figuras
rcParams.update({'font.size': 10})

def text_to_pdf(text, pdf, title='Texto'):
    """
    Cria uma figura de matplotlib e desenha 'text' nela,
    depois salva no PDF usando 'pdf.savefig()'.
    """
    # Tamanho da página em polegadas, por exemplo 8.5x11 (Letter) ou A4 (8.27x11.7)
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')  # remove eixos
    
    # Desenha o texto na coordenada (0,1) do sistema de eixos,
    # usando 'va=top' para começar de cima para baixo.
    ax.text(0.0, 1.0, text, fontsize=10, ha='left', va='top', fontfamily='monospace')
    
    ax.set_title(title, pad=10, fontsize=12)  # um título simples (opcional)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------
# Início do Script
# -----------------------------------------------------------
with PdfPages('relatorio_vendas_games.pdf') as pdf:

    # Buffer para capturar as impressões (prints)
    buffer = io.StringIO()
    
    # Vamos redirecionar tudo que seria impresso no console para 'buffer'
    with redirect_stdout(buffer):
        
        print("# SEÇÃO 1: Importando bibliotecas e lendo o CSV")
        print("Carregando arquivo CSV...")

        # Leitura do CSV
        df = pd.read_csv('data/Video+Game+Sales/vgchartz-2024.csv', encoding='utf-8')
        
        print("\n===== Primeiras linhas do dataset =====")
        print(df.head())

        print("\n===== Informações gerais =====")
        print(df.info())  # df.info() imprime no console, mas agora sai no buffer

        print("\n===== Quantidade de linhas e colunas =====")
        print(df.shape)

        print("\n===== Verificando valores nulos =====")
        print(df.isnull().sum())
        
        # SEÇÃO 2: Plot 1 (Histograma)
        if 'critic_score' in df.columns:
            print("\nGerando histograma de 'critic_score'...\n")
        
        # (Ao final do with, vamos transformar esse 'buffer' em string e criar uma figura para ele)

    # --------------------------------------------------------
    # 1) SALVAR NO PDF O TEXTO ACIMA
    # --------------------------------------------------------
    texto_inicial = buffer.getvalue()  # todo texto capturado
    # Criar uma figura com esse texto e salvar no PDF
    text_to_pdf(texto_inicial, pdf, title="Saída: Leitura e Info do Dataset")

    # Agora voltamos a imprimir normalmente (fora do redirect_stdout).
    # Se você quiser capturar outras partes, pode repetir a lógica acima.

    # SEÇÃO 2: Efetivamente criar a figura do histograma e salvá-la
    if 'critic_score' in df.columns:
        fig1 = plt.figure(figsize=(6,4))
        sns.histplot(data=df, x='critic_score', kde=True, color='blue')
        plt.title("Distribuição de critic_score")
        pdf.savefig(fig1)
        plt.close(fig1)

    # --------------------------------------------------------
    # SEÇÃO 3: Capturar mais textos (cálculo region_preference, etc.)
    # --------------------------------------------------------
    buffer2 = io.StringIO()
    with redirect_stdout(buffer2):
        print("SEÇÃO 3: Criando a coluna 'region_preference'\n")
        region_preferences = []
        for idx, row in df.iterrows():
            max_sales = max(row['na_sales'], row['jp_sales'], row['pal_sales'], row['other_sales'])
            if row['na_sales'] == max_sales:
                region_preferences.append('NA')
            elif row['jp_sales'] == max_sales:
                region_preferences.append('JP')
            elif row['pal_sales'] == max_sales:
                region_preferences.append('PAL')
            else:
                region_preferences.append('OTH')
        df['region_preference'] = region_preferences

        print("===== Distribuição da classe (region_preference) =====")
        print(df['region_preference'].value_counts())

    # Salva esse texto
    texto_region = buffer2.getvalue()
    text_to_pdf(texto_region, pdf, title="Saída: Region Preference")

    # Plot 2: Gráfico de contagem
    fig2 = plt.figure(figsize=(6,4))
    sns.countplot(x=df['region_preference'], palette='viridis')
    plt.title("Contagem de jogos por preferência regional")
    plt.xlabel("Região dominante")
    plt.ylabel("Quantidade de jogos")
    pdf.savefig(fig2)
    plt.close(fig2)

    # --------------------------------------------------------
    # SEÇÃO 4: Pré-processamento
    # --------------------------------------------------------
    buffer3 = io.StringIO()
    with redirect_stdout(buffer3):
        print("\nSEÇÃO 4: Pré-processamento\n")

        #Garantir que colunas que não são importantes não sejam utilizadas no treinamento

        df = df.drop(['img','title','release_date','last_update','total_sales'], axis=1, errors='ignore');

        # Contagem de linhas antes
        rows_before = df.shape[0]

        # Remover linhas com NaN em colunas que precisamos
        df.dropna(subset=['console','genre','publisher','developer',
                        'na_sales','jp_sales','pal_sales','other_sales',
                        'critic_score','region_preference'], 
                inplace=True)

        # Contagem de linhas depois
        rows_after = df.shape[0]
        rows_removed = rows_before - rows_after

        print(f"{rows_removed} linhas foram removidas por conterem valores nulos nas colunas necessárias.\n\n\n")


        #Trocando as variáveis que estão como string para dados números, já que modelos de machine learning normalmente utilizam este formato de dados.
        encoder = LabelEncoder()
        df['console'] = encoder.fit_transform(df['console'].astype(str))
        df['genre'] = encoder.fit_transform(df['genre'].astype(str))
        df['publisher'] = encoder.fit_transform(df['publisher'].astype(str))
        df['developer'] = encoder.fit_transform(df['developer'].astype(str))

        # Reimprimimos os dados para verificar se foi tratado corretamente

        print(df.head())
        print(df.info())

        # Imrpimindo a contagem por região para melhor visualização.
        sns.countplot(x=df['region_preference'], palette='viridis')
        plt.title("Contagem de jogos por preferência regional")


        y = df['region_preference']
        X = df.drop('region_preference', axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42
        )

        print("===== Formato dos dados =====")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")
        print("\nClasses no treino:")
        print(y_train.value_counts())
        print("\nClasses no teste:")
        print(y_test.value_counts())

    text_to_pdf(buffer3.getvalue(), pdf, title="Saída: Pré-processamento")

    # --------------------------------------------------------
    # SEÇÃO 5: Modelos (KNN, MLP, RF)
    # --------------------------------------------------------
    from sklearn.pipeline import Pipeline

    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    knn_pipeline.fit(X_train, y_train)
    pred_knn = knn_pipeline.predict(X_test)

    mlp_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42))
    ])
    mlp_pipeline.fit(X_train, y_train)
    pred_mlp = mlp_pipeline.predict(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    pred_rf = rf_model.predict(X_test)

    # --------------------------------------------------------
    # SEÇÃO 6: Avaliando e exibindo resultados
    # --------------------------------------------------------
    def avaliar_modelo(nome_modelo, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')

        # Em vez de printar direto, retornamos string formatada
        metrics_str = (
            f"=== {nome_modelo} ===\n"
            f"Acurácia : {acc:.4f}\n"
            f"Precisão : {prec:.4f}\n"
            f"Recall   : {rec:.4f}\n"
        )
        return metrics_str

    # Capturar cada resultado em uma string
    knn_str = avaliar_modelo("KNN", y_test, pred_knn)
    mlp_str = avaliar_modelo("MLP", y_test, pred_mlp)
    rf_str  = avaliar_modelo("RandomForest", y_test, pred_rf)

    # Matriz de confusão -> salvamos cada uma em PDF
    def plot_confusion(nome_modelo, y_true, y_pred):
        fig_cm, ax_cm = plt.subplots(figsize=(5,4))
        cm_display = ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            cmap=plt.get_cmap('Blues'),
            normalize='true',
            values_format='.2f',
            ax=ax_cm
        )
        ax_cm.set_title(f"Matriz de confusão - {nome_modelo}")
        pdf.savefig(fig_cm)
        plt.close(fig_cm)

    plot_confusion("KNN", y_test, pred_knn)
    plot_confusion("MLP", y_test, pred_mlp)
    plot_confusion("RandomForest", y_test, pred_rf)

    # DataFrame comparativo
    resultados = pd.DataFrame({
        'Algoritmo': ['KNN','MLP','RandomForest'],
        'Acuracia': [
            accuracy_score(y_test, pred_knn),
            accuracy_score(y_test, pred_mlp),
            accuracy_score(y_test, pred_rf)
        ],
        'Precisao': [
            precision_score(y_test, pred_knn, average='weighted'),
            precision_score(y_test, pred_mlp, average='weighted'),
            precision_score(y_test, pred_rf, average='weighted')
        ],
        'Recall': [
            recall_score(y_test, pred_knn, average='weighted'),
            recall_score(y_test, pred_mlp, average='weighted'),
            recall_score(y_test, pred_rf, average='weighted')
        ]
    })

    # Concatenar as métricas em uma string
    resultado_text = (
        "\n===== Comparativo final =====\n"
        f"{resultados}\n\n"
        f"{knn_str}\n{mlp_str}\n{rf_str}\n"
    )

    # Salvar esse texto (métricas e tabela) no PDF
    text_to_pdf(resultado_text, pdf, title="Saída: Resultados Finais")

    # Plot comparativo
    fig_comp = plt.figure(figsize=(6,4))
    sns.barplot(data=resultados, x='Algoritmo', y='Acuracia', palette='magma')
    plt.title("Comparação de Acurácia por Algoritmo")
    plt.ylim(0,1)
    pdf.savefig(fig_comp)
    plt.close(fig_comp)

print("\nPDF gerado com sucesso: relatorio_vendas_games.pdf")
