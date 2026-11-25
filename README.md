# Aplicativo: valor de shopping centers por transações 2020–2026

Aplicativo de demonstração em Streamlit para o curso IBAPE/SP de Inteligência Artificial na Engenharia de Avaliações. Ele reproduz o estudo do relatório "Valor de shopping centers por transações 2020–2026" com os mesmos dados, modelos e testes.

## O que o aplicativo mostra

| Aba | Conteúdo |
|---|---|
| Amostra | As 80 transações, com filtros por ano, localização e tipo de participação; gráficos VU × ABL e VU por ano; os 176 registros originais e os motivos de exclusão |
| Modelo e diagnóstico | Os dois modelos do estudo ou um modelo montado pelo usuário, estimado por mínimos quadrados ordinários ou ponderados, ridge, lasso ou Huber: equação, coeficientes, testes da NBR 14653-2, erros LOO e LOSO, gráficos de resíduos |
| Comparação de algoritmos | Os 30 modelos testados em validação cruzada aninhada, a revisão dos estimadores lineares para amostras pequenas e um teste rápido ao vivo com regressão, SVR, k-vizinhos, random forest e gradient boosting |
| Avaliar um shopping | Calculadora para shopping de qualquer cidade do país (localização classificada automaticamente), com os dois modelos, intervalo de confiança de 80%, campo de arbítrio de ±15%, valor arbitrado, grau de precisão, verificação de domínio e memória de cálculo |
| Método e conclusões | Resumo do método, dos achados e dos limites |

## Como abrir no Mac

1. Instale o Python 3.11 ou mais recente, de https://www.python.org/downloads/ (uma vez só).
2. Dê dois cliques em **Abrir aplicativo (Mac).command**. Na primeira vez o macOS pode pedir confirmação: clique com o botão direito no arquivo, escolha **Abrir** e confirme.
3. O script instala as bibliotecas na primeira execução e abre o aplicativo no navegador, em http://localhost:8501.
4. Para encerrar, feche a janela do Terminal.

Se o macOS disser que o arquivo não pode ser executado por falta de permissão, abra o Terminal e digite `chmod +x ` (com espaço no fim), arraste o arquivo para a janela, tecle Enter e tente de novo.

## Como abrir no Windows

1. Instale o Python 3.11 ou mais recente, de https://www.python.org/downloads/, marcando a opção **Add python.exe to PATH**.
2. Dê dois cliques em **Abrir aplicativo (Windows).bat**.

## Pelo terminal (qualquer sistema)

```
cd "pasta do aplicativo"
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Publicar na internet para os alunos

O Streamlit Community Cloud hospeda o aplicativo de graça:

1. Crie um repositório no GitHub e envie esta pasta inteira (app.py, requirements.txt e a pasta dados).
2. Entre em https://share.streamlit.io com a conta do GitHub, clique em **Create app**, escolha o repositório e o arquivo app.py.
3. Em poucos minutos o aplicativo fica disponível num endereço público que pode ser enviado aos alunos.

## Arquivos

- `app.py`: o aplicativo.
- `requirements.txt`: bibliotecas necessárias.
- `dados/transacoes.csv`: as 80 transações com todas as variáveis.
- `dados/base_completa.csv`: os 176 registros reunidos, com a decisão de inclusão.
- `dados/cidades.csv`: a localização (LOC) das cidades das regiões metropolitanas de SP e RJ e das cidades da amostra; qualquer outra cidade recebe LOC 0.
- `dados/revisao_linear.csv`: resultados da revisão dos estimadores lineares.
- `dados/comparacao_modelos.csv`: resultados da comparação dos 30 modelos.

Para acrescentar novas transações, inclua linhas em `dados/transacoes.csv` preenchendo também as colunas derivadas (grupo_shopping, vu_rs_m2, ctrl, p2026, dl e loc). O aplicativo reestima os modelos ao abrir. A aba de comparação dos 30 modelos usa resultados gravados e não se atualiza sozinha; o teste rápido ao vivo, sim.
