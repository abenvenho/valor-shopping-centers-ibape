"""
Valor de shopping centers por transações 2020–2026
Aplicativo de demonstração para o curso IBAPE/SP — Inteligência Artificial na Engenharia de Avaliações

Como rodar (no terminal, dentro desta pasta):
    pip install -r requirements.txt
    streamlit run app.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Valor de shopping centers", page_icon="🏬", layout="wide")

PASTA_DADOS = Path(__file__).parent / "dados"
ABL_REF = 36_800  # ABL média da amostra, ponto de referência do termo de interação
CORES_LOC = {0: "#9aa0a6", 1: "#8ab4f8", 2: "#1a73e8"}
NOMES_LOC = {0: "Demais cidades (LOC 0)", 1: "Rio e RM de SP e RJ (LOC 1)", 2: "Capital paulista (LOC 2)"}

# Variáveis disponíveis para o construtor de modelos: coluna -> rótulo
VARIAVEIS = {
    "ctrl": "CTRL — bloco de controle (participação ≥ 50%)",
    "ln_abl": "ln(ABL)",
    "p2026": "P2026 — transação em 2026",
    "dl": "DL — lojas por 1.000 m² de ABL",
    "loc": "LOC — localização (0, 1 ou 2)",
    "loc_x_ln_abl": "LOC × ln(ABL/36.800) — interação porte × localização",
    "ln_part": "ln(participação negociada)",
    "tempo_anos": "Tempo em anos desde jan/2020 (tendência)",
    "capital": "Capital estadual",
    "sp_capital": "Capital paulista (indicador)",
    "rm_sp_rj": "Região metropolitana de SP ou RJ (indicador)",
    "sudeste": "Região Sudeste",
    "idade_anos": "Idade do shopping (anos)",
    "outlet": "Outlet",
    "vagas_por_mil_m2": "Vagas por 1.000 m² de ABL (censo, 75 casos)",
    "area_constr_sobre_abl": "Área construída ÷ ABL (censo, 60 casos)",
    "pct_classe_a": "% de público classe A (censo, 55 casos)",
}
ROTULO_CURTO = {
    "const": "Constante", "ctrl": "CTRL", "ln_abl": "ln(ABL)", "p2026": "P2026", "dl": "DL", "loc": "LOC",
    "loc_x_ln_abl": "LOC × ln(ABL/36.800)", "ln_part": "ln(participação)", "tempo_anos": "Tempo",
    "capital": "Capital", "sp_capital": "Capital paulista", "rm_sp_rj": "RM SP/RJ", "sudeste": "Sudeste",
    "idade_anos": "Idade", "outlet": "Outlet", "vagas_por_mil_m2": "Vagas/1.000 m²",
    "area_constr_sobre_abl": "Área constr./ABL", "pct_classe_a": "% classe A",
}
MODELOS = {
    "Recomendado — interação ABL × LOC": ["ctrl", "p2026", "dl", "loc", "loc_x_ln_abl"],
    "Referência — 5 variáveis": ["ctrl", "ln_abl", "p2026", "dl", "loc"],
}
# Conjunto de 13 variáveis usado no teste rápido de algoritmos
VARS_13 = ["ln_part", "ctrl", "ln_abl", "tempo_anos", "p2026", "dl", "loc",
           "capital", "sp_capital", "rm_sp_rj", "sudeste", "idade_anos", "outlet"]

NOMES_LINEARES = {
    "OLS_M5i": "OLS, 5 variáveis do recomendado", "OLS_A5": "OLS, 5 variáveis da referência", "OLS_POOL": "OLS, 14 candidatas",
    "BestBIC_POOL": "Melhor subconjunto por BIC (até 8 de 14)", "BMA_POOL": "Média bayesiana de modelos (14 candidatas)",
    "WLS_LOC": "Ponderado: variância por LOC", "WLS_FGLS": "Ponderado: função de variância (FGLS)",
    "WLS_shopping": "Ponderado: 1 ÷ transações do shopping", "WLS_part": "Ponderado: √participação", "WLS_recencia": "Ponderado: recência",
    "Ridge_M5i": "Ridge, 5 variáveis", "Lasso_M5i": "Lasso, 5 variáveis", "ENet_M5i": "Elastic net, 5 variáveis",
    "Ridge_POOL": "Ridge, 14 candidatas", "Lasso_POOL": "Lasso, 14 candidatas", "ENet_POOL": "Elastic net, 14 candidatas",
    "AdaLasso_POOL": "Lasso adaptativo, 14 candidatas", "BayesRidge_M5i": "Ridge bayesiana, 5 variáveis",
    "BayesRidge_POOL": "Ridge bayesiana, 14 candidatas", "ARD_POOL": "Bayesiana ARD, 14 candidatas", "Huber_M5i": "Huber, 5 variáveis",
    "RLMTukey_M5i": "Tukey biquadrada, 5 variáveis", "LAD_M5i": "Mínimos desvios absolutos, 5 variáveis", "TheilSen_M5i": "Theil-Sen, 5 variáveis",
    "PLS_POOL": "PLS, 14 candidatas", "PCR_POOL": "Componentes principais, 14 candidatas", "Mixed_M5i": "Misto por shopping, 5 variáveis",
    "Bagging_M5i": "OLS com bagging, 5 variáveis",
}

NOMES_ALGORITMOS = {
    "OLS_A5": "Regressão de referência (5 var.)", "OLS_M5i": "Regressão com interação ABL × LOC",
    "OLS_A6": "Referência + interação (6 var.)", "Sel1_A5int": "Referência + 1 termo escolhido na dobra",
    "Sel2_A5int": "Referência + até 2 termos escolhidos na dobra", "Sel3_A5int": "Referência + até 3 termos escolhidos na dobra",
    "OLS_Poly2_A5": "Polinomial grau 2 (OLS)", "Ridge_Poly2_A5": "Polinomial grau 2 (ridge)",
    "Ridge_Poly2_B": "Polinomial grau 2, 13 var. (ridge)", "Lasso_Poly2_B": "Polinomial grau 2, 13 var. (lasso)",
    "ENet_B": "Elastic net, 13 var.", "OLS_B": "Regressão com 13 var.", "BestSubset_BIC": "Seleção automática refeita na dobra",
    "Huber_A5": "Regressão robusta de Huber", "Mixed_A5": "Efeitos mistos por shopping", "Spline_A5": "Splines + ridge",
    "Spline_B": "Splines + ridge, 13 var.", "GAM_A5": "GAM (splines penalizados)", "kNN_A5": "k-vizinhos",
    "kNN_B": "k-vizinhos, 13 var.", "SVR_A5": "SVR kernel RBF", "SVR_B": "SVR kernel RBF, 13 var.",
    "GP_A5": "Processo gaussiano", "GP_B": "Processo gaussiano, 13 var.", "RF_A5": "Random forest, 5 var.",
    "RF_B": "Random forest, 13 var.", "RF_C": "Random forest, 19 var.", "ExtraTrees_B": "Extra trees, 13 var.",
    "GBM_A5": "Gradient boosting, 5 var.", "GBM_B": "Gradient boosting, 13 var.", "XGB_B": "XGBoost, 13 var.",
    "LGBM_B": "LightGBM, 13 var.", "HGB_C": "Histogram gradient boosting, 19 var.", "MLP_B": "Rede neural (MLP), 13 var.",
    "Hybrid_OLS+RF": "Regressão + random forest nos resíduos", "Ens_OLS+SVR": "Média: regressão e SVR",
    "Ens_OLS+GP+RF": "Média: regressão, proc. gaussiano e RF",
}


# ----------------------------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------------------------
def br(x, casas=0):
    """Formata número no padrão brasileiro (1.234,56)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, casas=0):
    return br(100 * x, casas) + "%"


def grau_precisao(amplitude):
    """NBR 14653-2: amplitude do IC de 80% em torno do valor central."""
    if amplitude <= 0.30:
        return "III"
    if amplitude <= 0.40:
        return "II"
    if amplitude <= 0.50:
        return "I"
    return "fora dos graus"


def normalizar(texto):
    """Maiúsculas, sem acentos e sem espaços extras, para comparar nomes de cidades."""
    import unicodedata
    t = unicodedata.normalize("NFD", str(texto).strip().upper())
    return " ".join("".join(c for c in t if unicodedata.category(c) != "Mn").split())


def classificar_loc(cidade, uf, cidades):
    """LOC = 2 na capital paulista; 1 no Rio de Janeiro capital e nas RMs de SP e RJ; 0 em qualquer outra cidade."""
    chave = normalizar(cidade) + "|" + str(uf).upper()
    achou = cidades[cidades["chave"] == chave]
    if len(achou):
        return int(achou["loc"].iloc[0]), achou["criterio"].iloc[0]
    return 0, "Fora das regiões metropolitanas de São Paulo e do Rio de Janeiro"


@st.cache_data
def carregar_dados():
    t = pd.read_csv(PASTA_DADOS / "transacoes.csv")
    t["ln_vu"] = np.log(t["vu_rs_m2"])
    t["ln_abl"] = np.log(t["abl_m2"])
    t["ln_part"] = np.log(t["participacao"])
    t["loc_x_ln_abl"] = t["loc"] * np.log(t["abl_m2"] / ABL_REF)
    t["localizacao"] = t["loc"].map(NOMES_LOC)
    base = pd.read_csv(PASTA_DADOS / "base_completa.csv")
    cidades = pd.read_csv(PASTA_DADOS / "cidades.csv")
    comparacao = pd.read_csv(PASTA_DADOS / "comparacao_modelos.csv")
    linear = pd.read_csv(PASTA_DADOS / "revisao_linear.csv")
    cidades["chave"] = (cidades["cidade"].map(normalizar) + "|" + cidades["uf"].str.upper())
    return t, base, cidades, comparacao, linear


def pesos_por_loc(y, X, loc):
    """Mínimos quadrados ponderados em dois passos: peso = 1 / variância residual do OLS em cada nível de LOC."""
    r = sm.OLS(y, X).fit()
    var = pd.Series(r.resid ** 2).groupby(np.asarray(loc)).mean()
    return 1.0 / pd.Series(np.asarray(loc)).map(var).values


def _ajusta(y, X, loc, estimador):
    if estimador == "WLS":
        return sm.WLS(y, X, weights=pesos_por_loc(y, X, loc)).fit()
    return sm.OLS(y, X).fit()


@st.cache_data
def ajustar_modelo(dados, variaveis, estimador="OLS"):
    """Regressão por mínimos quadrados (ordinários ou ponderados) com diagnóstico completo e validação cruzada."""
    variaveis = list(variaveis)
    d = dados.dropna(subset=variaveis).reset_index(drop=True)
    y = d["ln_vu"].values
    X = sm.add_constant(d[variaveis].astype(float))
    loc = d["loc"].values
    res = _ajusta(y, X, loc, estimador)
    if estimador == "WLS":
        res_cl = sm.WLS(y, X, weights=res.model.weights).fit(cov_type="cluster", cov_kwds={"groups": d["grupo_shopping"]})
    else:
        res_cl = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["grupo_shopping"]})

    # Validação cruzada deixando uma transação de fora (LOO) e deixando o shopping inteiro de fora (LOSO).
    # Nos mínimos quadrados ponderados, os pesos são reestimados só com os dados de treino de cada rodada.
    pred_loo = np.zeros(len(y))
    for i in range(len(y)):
        treino = np.arange(len(y)) != i
        pred_loo[i] = np.asarray(_ajusta(y[treino], X[treino], loc[treino], estimador).predict(X.iloc[[i]]))[0]
    pred_loso = np.zeros(len(y))
    for g in d["grupo_shopping"].unique():
        teste = (d["grupo_shopping"] == g).values
        pred_loso[teste] = np.asarray(_ajusta(y[~teste], X[~teste], loc[~teste], estimador).predict(X[teste]))

    if estimador == "WLS":
        # Influência com o desenho ponderado: h = w·x(XᵀWX)⁻¹xᵀ
        w = res.model.weights
        Xw = X.values * np.sqrt(w)[:, None]
        h = np.einsum("ij,jk,ik->i", Xw, np.linalg.inv(Xw.T @ Xw), Xw)
        resid_pad = res.wresid / np.sqrt(res.scale * (1 - h))
        cook = resid_pad ** 2 * h / (X.shape[1] * (1 - h))
    else:
        infl = res.get_influence()
        cook = infl.cooks_distance[0]
        resid_pad = infl.resid_studentized_internal
    resid_w = res.wresid if estimador == "WLS" else res.resid
    vif = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])] if X.shape[1] > 2 else [1.0]

    coef = pd.DataFrame({
        "Variável": [ROTULO_CURTO.get(c, c) for c in X.columns],
        "Coeficiente": res.params.values,
        "Erro-padrão": res.bse.values,
        "t": res.tvalues.values,
        "p": res.pvalues.values,
        "p agrupado por shopping": res_cl.pvalues.values,
        "Efeito sobre o VU": [np.nan] + list(np.exp(res.params.values[1:]) - 1),
    })
    diag = {
        "n": len(y), "k": len(variaveis), "R²": res.rsquared, "R² ajustado": res.rsquared_adj,
        "F": res.fvalue, "p do F": res.f_pvalue, "s": np.sqrt(sm.OLS(y, X).fit().scale) if estimador == "WLS" else np.sqrt(res.scale),
        "BIC": res.bic,
        "RMSE LOO": np.sqrt(np.mean((y - pred_loo) ** 2)),
        "RMSE LOSO": np.sqrt(np.mean((y - pred_loso) ** 2)),
        "Erro absoluto mediano LOO": np.median(np.abs(np.exp(y - pred_loo) - 1)),
        "Shapiro-Wilk p": stats.shapiro(resid_w).pvalue,
        "Jarque-Bera p": jarque_bera(resid_w)[1],
        "Breusch-Pagan p": het_breuschpagan(resid_w, X)[1],
        "Durbin-Watson": durbin_watson(resid_w),
        "VIF máximo": max(vif), "Cook máximo": cook.max(),
        "Resíduos em ±1,96": np.mean(np.abs(resid_pad) <= 1.96),
        "Maior p dos regressores": res.pvalues.values[1:].max(),
    }
    saida = d[["id", "shopping", "ano", "localizacao", "loc", "abl_m2", "vu_rs_m2"]].copy()
    saida["vu_estimado"] = np.exp(res.fittedvalues.values)
    saida["vu_previsto_loso"] = np.exp(pred_loso)
    saida["residuo"] = res.resid
    saida["residuo_padronizado"] = resid_pad
    saida["cook"] = cook
    return {"res": res, "coef": coef, "diag": diag, "saida": saida, "variaveis": variaveis}


@st.cache_data
def ajustar_penalizado(dados, variaveis, estimador):
    """Ridge, Lasso ou Huber com variáveis padronizadas; penalização escolhida por validação cruzada agrupada por shopping."""
    from sklearn.linear_model import HuberRegressor, Lasso, Ridge
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    variaveis = list(variaveis)
    d = dados.dropna(subset=variaveis).reset_index(drop=True)
    y = d["ln_vu"].values
    X = d[variaveis].astype(float).values
    grupos = d["grupo_shopping"].values
    if estimador == "Ridge":
        base, grade, passo = Ridge(), {"ridge__alpha": np.logspace(-3, 3, 40)}, "ridge"
    elif estimador == "Lasso":
        base, grade, passo = Lasso(max_iter=50_000), {"lasso__alpha": np.logspace(-4, 0, 40)}, "lasso"
    else:
        base, grade, passo = HuberRegressor(alpha=0, max_iter=1000), {"huberregressor__epsilon": [1.2, 1.35, 1.6, 2.0]}, "huberregressor"
    busca = GridSearchCV(make_pipeline(StandardScaler(), base), grade, cv=GroupKFold(5), scoring="neg_mean_squared_error")
    busca.fit(X, y, groups=grupos)
    melhor = busca.best_params_
    fabrica = lambda: make_pipeline(StandardScaler(), base.__class__(**{**base.get_params(), **{k.split("__")[1]: v for k, v in melhor.items()}}))
    modelo = busca.best_estimator_
    sc, est = modelo.named_steps["standardscaler"], modelo.named_steps[passo]
    b = est.coef_ / sc.scale_
    a = est.intercept_ - np.sum(b * sc.mean_)
    pred_loo = np.zeros(len(y))
    for i in range(len(y)):
        treino = np.arange(len(y)) != i
        pred_loo[i] = fabrica().fit(X[treino], y[treino]).predict(X[[i]])[0]
    pred_loso = np.zeros(len(y))
    for g in np.unique(grupos):
        teste = grupos == g
        pred_loso[teste] = fabrica().fit(X[~teste], y[~teste]).predict(X[teste])
    ajuste = modelo.predict(X)
    coef = pd.DataFrame({"Variável": ["Constante"] + [ROTULO_CURTO.get(c, c) for c in variaveis],
                         "Coeficiente": np.r_[a, b], "Efeito sobre o VU": [np.nan] + list(np.exp(b) - 1)})
    diag = {"n": len(y), "R²": 1 - np.sum((y - ajuste) ** 2) / np.sum((y - y.mean()) ** 2),
            "RMSE LOO": np.sqrt(np.mean((y - pred_loo) ** 2)), "RMSE LOSO": np.sqrt(np.mean((y - pred_loso) ** 2)),
            "Erro absoluto mediano LOO": np.median(np.abs(np.exp(y - pred_loo) - 1)),
            "Parâmetro": ", ".join(f"{k.split('__')[1]} = {v:.4g}" for k, v in melhor.items())}
    saida = d[["id", "shopping", "ano", "localizacao", "loc", "abl_m2", "vu_rs_m2"]].copy()
    saida["vu_previsto_loso"] = np.exp(pred_loso)
    return {"coef": coef, "diag": diag, "saida": saida}


def prever(res, variaveis, entrada):
    """Previsão em log com intervalos de confiança e de predição de 80%."""
    x = pd.DataFrame([[1.0] + [entrada[v] for v in variaveis]], columns=["const"] + variaveis)
    q = res.get_prediction(x).summary_frame(alpha=0.20)
    centro = float(q["mean"].iloc[0])
    return {
        "vu": np.exp(centro),
        "ic": (np.exp(float(q["mean_ci_lower"].iloc[0])), np.exp(float(q["mean_ci_upper"].iloc[0]))),
        "ip": (np.exp(float(q["obs_ci_lower"].iloc[0])), np.exp(float(q["obs_ci_upper"].iloc[0]))),
    }


def equacao_latex(res):
    termos = []
    for nome, b in res.params.items():
        rot = {"const": "", "ctrl": r"\,CTRL", "ln_abl": r"\,\ln(ABL)", "p2026": r"\,P2026", "dl": r"\,DL",
               "loc": r"\,LOC", "loc_x_ln_abl": r"\,LOC\cdot\ln(ABL/36.800)"}.get(nome, r"\,\text{" + ROTULO_CURTO.get(nome, nome).replace("%", r"\%") + "}")
        valor = br(abs(b), 3).replace(",", "{,}")
        sinal = "-" if b < 0 else "+"
        termos.append((sinal, valor + rot))
    s = r"\ln(VU) = " + ("-" if termos[0][0] == "-" else "") + termos[0][1]
    for sinal, t in termos[1:]:
        s += f" {sinal} {t}"
    return s


dados, base_completa, cidades, comparacao, revisao_linear = carregar_dados()

# ----------------------------------------------------------------------------------------------
# Cabeçalho
# ----------------------------------------------------------------------------------------------
st.title("Valor de shopping centers por transações 2020–2026")
st.caption("Método comparativo direto com inferência estatística · 80 transações de participações em 60 shoppings · "
           "aplicativo de demonstração para o curso IBAPE/SP de Inteligência Artificial")

with st.sidebar:
    st.header("Sobre")
    st.markdown(
        "**VU** = valor da transação ÷ (participação × ABL), em R\\$/m² de ABL, referido a 100% do shopping.\n\n"
        "**Modelo recomendado:** regressão log-linear com interação entre porte e localização.\n\n"
        "**Referência:** regressão de cinco variáveis com ln(ABL).\n\n"
        "Cap rate não foi usado como variável. Pacotes de vários shoppings foram excluídos."
    )
    st.divider()
    st.markdown("**Localização (LOC)**\n\n0 — demais cidades\n\n1 — Rio de Janeiro capital e regiões metropolitanas de SP e RJ fora da capital paulista\n\n2 — capital paulista")
    st.divider()
    st.caption("Estudo de mercado; não substitui laudo de avaliação. Use dentro do domínio da amostra.")

aba_amostra, aba_modelo, aba_algoritmos, aba_avaliar, aba_metodo = st.tabs(
    ["📋 Amostra", "📈 Modelo e diagnóstico", "🤖 Comparação de algoritmos", "🏬 Avaliar um shopping", "📘 Método e conclusões"]
)

# ----------------------------------------------------------------------------------------------
# 1. Amostra
# ----------------------------------------------------------------------------------------------
with aba_amostra:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transações no modelo", len(dados))
    c2.metric("Shoppings", dados["grupo_shopping"].nunique())
    c3.metric("Cidades", dados["cidade"].nunique())
    c4.metric("Período", f"{dados['data'].min()[:4]}–{dados['data'].max()[:4]}")
    c5.metric("Valor negociado", f"R$ {br(dados['valor_rs'].sum() / 1e9, 1)} bi")

    f1, f2, f3 = st.columns(3)
    anos = f1.multiselect("Ano", sorted(dados["ano"].unique()), default=sorted(dados["ano"].unique()))
    locs = f2.multiselect("Localização", [0, 1, 2], default=[0, 1, 2], format_func=lambda v: NOMES_LOC[v])
    tipo = f3.radio("Participação", ["Todas", "Frações minoritárias", "Blocos de controle"], horizontal=True)
    filtro = dados["ano"].isin(anos) & dados["loc"].isin(locs)
    if tipo == "Frações minoritárias":
        filtro &= dados["ctrl"] == 0
    elif tipo == "Blocos de controle":
        filtro &= dados["ctrl"] == 1
    df = dados[filtro]

    g1, g2 = st.columns(2)
    fig = px.scatter(df, x="abl_m2", y="vu_rs_m2", color="localizacao", log_x=True, log_y=True,
                     color_discrete_map={v: CORES_LOC[k] for k, v in NOMES_LOC.items()},
                     hover_data={"shopping": True, "ano": True, "participacao": ":.1%", "abl_m2": ":,.0f", "vu_rs_m2": ":,.0f", "localizacao": False},
                     labels={"abl_m2": "ABL (m²)", "vu_rs_m2": "VU (R$/m²)", "localizacao": ""},
                     title="VU × ABL: o porte só pesa nas metrópoles de SP e RJ")
    fig.update_layout(legend=dict(orientation="h", y=-0.2), height=430)
    g1.plotly_chart(fig, width="stretch")
    fig = px.strip(df, x="ano", y="vu_rs_m2", color="localizacao", log_y=True,
                   color_discrete_map={v: CORES_LOC[k] for k, v in NOMES_LOC.items()},
                   hover_data=["shopping"], labels={"ano": "Ano", "vu_rs_m2": "VU (R$/m²)", "localizacao": ""},
                   title="VU por ano da transação")
    medianas = df.groupby("ano")["vu_rs_m2"].median().reset_index()
    fig.add_trace(go.Scatter(x=medianas["ano"], y=medianas["vu_rs_m2"], mode="markers+lines", name="Mediana",
                             marker=dict(symbol="line-ew-open", size=30, color="black"), line=dict(color="black", width=1, dash="dot")))
    fig.update_layout(legend=dict(orientation="h", y=-0.2), height=430)
    g2.plotly_chart(fig, width="stretch")

    st.subheader(f"Transações selecionadas: {len(df)}")
    colunas = ["data", "shopping", "cidade", "uf", "participacao", "valor_rs", "abl_m2", "vu_rs_m2", "lojas_dl", "dl", "loc", "fonte", "ressalva"]
    st.dataframe(df[colunas], hide_index=True, width="stretch", column_config={
        "data": "Data", "shopping": "Shopping", "cidade": "Cidade", "uf": "UF",
        "participacao": st.column_config.NumberColumn("Participação", format="percent"),
        "valor_rs": st.column_config.NumberColumn("Valor (R$)", format="localized"),
        "abl_m2": st.column_config.NumberColumn("ABL (m²)", format="localized"),
        "vu_rs_m2": st.column_config.NumberColumn("VU (R$/m²)", format="localized"),
        "lojas_dl": "Lojas", "dl": st.column_config.NumberColumn("Lojas/1.000 m²", format="%.2f"),
        "loc": "LOC", "fonte": "Fonte", "ressalva": "Ressalva"})
    st.download_button("Baixar a amostra (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "amostra_shoppings.csv", "text/csv")

    with st.expander("De 176 registros a 80 transações: critérios de exclusão"):
        excl = base_completa[base_completa["incluida"] == "Não"].copy()
        def categoria(m):
            m = str(m)
            if m.startswith("Pacote"):
                return "Pacote com vários shoppings e valor único"
            if m.startswith("MoU"):
                return "Memorando sem valor ou pacote"
            if m.startswith("Duplicata") or m.startswith("Agregada"):
                return "Duplicata ou venda agregada a outra linha"
            if "não informada" in m:
                return "Participação não informada"
            if "não divulgado" in m:
                return "Valor não divulgado"
            if "não concretizada" in m:
                return "Operação não concretizada"
            return "Ativo em construção ou que não é shopping"
        excl["Categoria"] = excl["motivo_exclusao"].map(categoria)
        st.dataframe(excl["Categoria"].value_counts().rename("Registros").reset_index().rename(columns={"index": "Categoria"}),
                     hide_index=True, width="stretch")
        st.dataframe(base_completa, hide_index=True, width="stretch", height=300)

# ----------------------------------------------------------------------------------------------
# 2. Modelo e diagnóstico
# ----------------------------------------------------------------------------------------------
with aba_modelo:
    escolha = st.radio("Modelo", list(MODELOS) + ["Personalizado — escolha as variáveis"], horizontal=True)
    if escolha.startswith("Personalizado"):
        variaveis = st.multiselect("Variáveis independentes", list(VARIAVEIS), default=MODELOS["Referência — 5 variáveis"],
                                   format_func=lambda v: VARIAVEIS[v])
        st.caption("Experimente acrescentar variáveis do censo ou trocar ln(ABL) pela interação, e veja o que acontece com o erro LOSO.")
    else:
        variaveis = MODELOS[escolha]

    ESTIMADORES = {
        "Mínimos quadrados ordinários (OLS)": "OLS",
        "Mínimos quadrados ponderados (variância por LOC)": "WLS",
        "Ridge": "Ridge",
        "Lasso": "Lasso",
        "Huber (robusto)": "Huber",
    }
    estimador = st.radio("Estimador", list(ESTIMADORES), horizontal=True,
                         help="OLS é o estimador adotado. Os demais servem para verificar a robustez do modelo; veja a aba Comparação de algoritmos.")
    tipo = ESTIMADORES[estimador]

    if not variaveis:
        st.info("Escolha ao menos uma variável.")
    elif tipo in ("Ridge", "Lasso", "Huber"):
        with st.spinner("Escolhendo a penalização e validando..."):
            p = ajustar_penalizado(dados, tuple(variaveis), tipo)
        dg = p["diag"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("n", dg["n"])
        k2.metric("R²", br(dg["R²"], 3))
        k3.metric("Erro LOO (log)", br(dg["RMSE LOO"], 3))
        k4.metric("Erro LOSO (log)", br(dg["RMSE LOSO"], 3), help="Cada shopping previsto sem nenhuma transação dele no ajuste")
        k5.metric("Parâmetro escolhido", dg["Parâmetro"])
        ols = ajustar_modelo(dados, tuple(variaveis), "OLS")["diag"]
        st.caption(f"Com as mesmas variáveis, o OLS tem erro LOO {br(ols['RMSE LOO'], 3)} e LOSO {br(ols['RMSE LOSO'], 3)}. "
                   "Ridge e lasso encolhem os coeficientes para reduzir a variância; Huber reduz o peso dos resíduos grandes. "
                   "Os testes de inferência da NBR 14653-2 valem para o OLS, por isso não são mostrados aqui.")
        st.dataframe(p["coef"], hide_index=True, width="stretch", column_config={
            "Coeficiente": st.column_config.NumberColumn(format="%.4f"),
            "Efeito sobre o VU": st.column_config.NumberColumn(format="percent")})
        s = p["saida"]
        fig = px.scatter(s, x="vu_previsto_loso", y="vu_rs_m2", color="localizacao", log_x=True, log_y=True,
                         color_discrete_map={v: CORES_LOC[k] for k, v in NOMES_LOC.items()}, hover_data=["shopping", "ano"],
                         labels={"vu_previsto_loso": "VU previsto sem o próprio shopping (R$/m²)", "vu_rs_m2": "VU observado (R$/m²)", "localizacao": ""},
                         title=f"Observado × previsto fora da amostra (LOSO), {estimador}")
        lim = [min(s["vu_rs_m2"].min(), s["vu_previsto_loso"].min()) * 0.9, max(s["vu_rs_m2"].max(), s["vu_previsto_loso"].max()) * 1.1]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="gray", dash="dash"), name="45°"))
        fig.update_layout(legend=dict(orientation="h", y=-0.25), height=450)
        st.plotly_chart(fig, width="stretch")
    else:
        m = ajustar_modelo(dados, tuple(variaveis), tipo)
        dg = m["diag"]
        if tipo == "WLS":
            st.caption("Pesos = 1 ÷ variância residual do OLS em cada nível de LOC, reestimados em cada rodada de validação. "
                       "Os testes de resíduos usam os resíduos ponderados; o desvio-padrão mostrado é o do OLS, para comparação.")
        st.latex(equacao_latex(m["res"]))
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("n", dg["n"])
        k2.metric("R² ajustado", br(dg["R² ajustado"], 3))
        k3.metric("Desvio-padrão (log)", br(dg["s"], 3))
        k4.metric("Erro LOO (log)", br(dg["RMSE LOO"], 3))
        k5.metric("Erro LOSO (log)", br(dg["RMSE LOSO"], 3), help="Cada shopping previsto sem nenhuma transação dele no ajuste")
        k6.metric("BIC", br(dg["BIC"], 1))
        if dg["n"] < len(dados):
            st.warning(f"{len(dados) - dg['n']} transações ficaram fora por falta de dado nas variáveis escolhidas; "
                       "compare BIC e erros só entre modelos com o mesmo n.")

        st.subheader("Coeficientes")
        st.dataframe(m["coef"], hide_index=True, width="stretch", column_config={
            "Coeficiente": st.column_config.NumberColumn(format="%.4f"),
            "Erro-padrão": st.column_config.NumberColumn(format="%.4f"),
            "t": st.column_config.NumberColumn(format="%.2f"),
            "p": st.column_config.NumberColumn(format="%.4f"),
            "p agrupado por shopping": st.column_config.NumberColumn(format="%.4f"),
            "Efeito sobre o VU": st.column_config.NumberColumn(format="percent", help="exp(coeficiente) − 1, por unidade da variável")})

        st.subheader("Testes")
        testes = pd.DataFrame([
            ("Significância dos regressores (NBR, grau III: ≤ 10%)", f"maior p = {br(dg['Maior p dos regressores'], 4)}", dg["Maior p dos regressores"] <= 0.10),
            ("Significância do modelo, teste F (grau III: ≤ 1%)", f"F = {br(dg['F'], 1)}; p = {dg['p do F']:.1e}", dg["p do F"] <= 0.01),
            ("Normalidade, Shapiro-Wilk", f"p = {br(dg['Shapiro-Wilk p'], 3)}", dg["Shapiro-Wilk p"] > 0.05),
            ("Normalidade, Jarque-Bera", f"p = {br(dg['Jarque-Bera p'], 3)}", dg["Jarque-Bera p"] > 0.05),
            ("Resíduos padronizados em ±1,96 (esperado ≈ 95%)", pct(dg["Resíduos em ±1,96"]), dg["Resíduos em ±1,96"] >= 0.90),
            ("Homocedasticidade, Breusch-Pagan", f"p = {br(dg['Breusch-Pagan p'], 3)}", dg["Breusch-Pagan p"] > 0.05),
            ("Autocorrelação em ordem cronológica, Durbin-Watson", br(dg["Durbin-Watson"], 2), 1.5 <= dg["Durbin-Watson"] <= 2.5),
            ("Multicolinearidade, VIF máximo", br(dg["VIF máximo"], 2), dg["VIF máximo"] < 5),
            ("Pontos influentes, distância de Cook máxima", br(dg["Cook máximo"], 3), dg["Cook máximo"] < 1),
        ], columns=["Teste", "Resultado", "Passa"])
        testes["Passa"] = testes["Passa"].map({True: "✅", False: "⚠️"})
        st.dataframe(testes, hide_index=True, width="stretch")

        s = m["saida"]
        h1, h2 = st.columns(2)
        fig = px.scatter(s, x="vu_previsto_loso", y="vu_rs_m2", color="localizacao", log_x=True, log_y=True,
                         color_discrete_map={v: CORES_LOC[k] for k, v in NOMES_LOC.items()}, hover_data=["shopping", "ano"],
                         labels={"vu_previsto_loso": "VU previsto sem o próprio shopping (R$/m²)", "vu_rs_m2": "VU observado (R$/m²)", "localizacao": ""},
                         title="Observado × previsto fora da amostra (LOSO)")
        lim = [min(s["vu_rs_m2"].min(), s["vu_previsto_loso"].min()) * 0.9, max(s["vu_rs_m2"].max(), s["vu_previsto_loso"].max()) * 1.1]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="gray", dash="dash"), name="45°"))
        fig.update_layout(legend=dict(orientation="h", y=-0.25), height=430)
        h1.plotly_chart(fig, width="stretch")
        fig = px.scatter(s, x=np.log(s["vu_estimado"]), y="residuo_padronizado", color="localizacao",
                         color_discrete_map={v: CORES_LOC[k] for k, v in NOMES_LOC.items()}, hover_data=["shopping", "ano"],
                         labels={"x": "ln(VU) estimado", "residuo_padronizado": "Resíduo padronizado", "localizacao": ""},
                         title="Resíduos padronizados × valores estimados")
        for yv in (-1.96, 0, 1.96):
            fig.add_hline(y=yv, line_dash="dot" if yv else "solid", line_color="gray")
        fig.update_layout(legend=dict(orientation="h", y=-0.25), height=430)
        h2.plotly_chart(fig, width="stretch")

        h3, h4 = st.columns(2)
        teor, obs = stats.probplot(s["residuo_padronizado"], dist="norm", fit=False)
        fig = go.Figure([go.Scatter(x=teor, y=obs, mode="markers", name="Resíduos", marker=dict(color="#1a73e8")),
                         go.Scatter(x=[-3, 3], y=[-3, 3], mode="lines", line=dict(color="gray", dash="dash"), name="Normal")])
        fig.update_layout(title="Gráfico de probabilidade normal", xaxis_title="Quantil teórico", yaxis_title="Resíduo padronizado", height=380, showlegend=False)
        h3.plotly_chart(fig, width="stretch")
        fig = px.histogram(s, x="residuo_padronizado", nbins=16, labels={"residuo_padronizado": "Resíduo padronizado"},
                           title="Distribuição dos resíduos padronizados")
        fig.update_layout(height=380, yaxis_title="Transações")
        h4.plotly_chart(fig, width="stretch")

        with st.expander("Tabela de resíduos e previsões"):
            st.dataframe(s.sort_values("residuo_padronizado"), hide_index=True, width="stretch", column_config={
                "abl_m2": st.column_config.NumberColumn("ABL (m²)", format="localized"),
                "vu_rs_m2": st.column_config.NumberColumn("VU observado", format="localized"),
                "vu_estimado": st.column_config.NumberColumn("VU estimado", format="localized"),
                "vu_previsto_loso": st.column_config.NumberColumn("VU previsto LOSO", format="localized"),
                "residuo": st.column_config.NumberColumn("Resíduo (log)", format="%.3f"),
                "residuo_padronizado": st.column_config.NumberColumn("Resíduo padronizado", format="%.2f"),
                "cook": st.column_config.NumberColumn("Cook", format="%.3f")})

# ----------------------------------------------------------------------------------------------
# 3. Comparação de algoritmos
# ----------------------------------------------------------------------------------------------
with aba_algoritmos:
    st.markdown(
        "Trinta modelos foram comparados em **validação cruzada aninhada**: os hiperparâmetros foram ajustados só dentro de "
        "cada dobra de treino, com partição por shopping. **LOO** deixa uma transação de fora; **LOSO** deixa de fora todas as "
        "transações do mesmo shopping, o que reproduz a avaliação de um shopping novo. O intervalo de 95% da diferença de erro "
        "vem de 4.000 reamostragens por shopping."
    )
    esquema = st.radio("Esquema de validação", ["LOSO", "LOO"], horizontal=True)
    c = comparacao[comparacao["esquema"] == esquema].copy()
    c["Modelo"] = c["modelo"].map(NOMES_ALGORITMOS).fillna(c["modelo"])
    c = c[c["modelo"] != "OLS_A5"].sort_values("d_rmse", ascending=False)
    c["Resultado"] = np.where(c["d_hi"] < 0, "Melhor que a referência", np.where(c["d_lo"] > 0, "Pior que a referência", "Sem diferença significativa"))
    ref = comparacao[(comparacao["esquema"] == esquema) & (comparacao["modelo"] == "OLS_A5")]["rmse"].iloc[0]
    c["erro_mais"] = c["d_hi"] - c["d_rmse"]
    c["erro_menos"] = c["d_rmse"] - c["d_lo"]
    fig = px.scatter(c, x="d_rmse", y="Modelo", color="Resultado",
                     color_discrete_map={"Melhor que a referência": "#1a73e8", "Pior que a referência": "#d93025", "Sem diferença significativa": "#9aa0a6"},
                     error_x="erro_mais", error_x_minus="erro_menos",
                     hover_data={"rmse": ":.3f", "d_rmse": ":+.3f"},
                     labels={"d_rmse": f"Diferença de RMSE (log) em relação à regressão de referência (RMSE {br(ref, 3)})"},
                     title=f"Erro de cada modelo menos o erro da regressão de referência, {esquema}")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=900, legend=dict(orientation="h", y=-0.08), yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(c[["Modelo", "rmse", "d_rmse", "d_lo", "d_hi", "mdape", "w25", "Resultado"]].sort_values("rmse"),
                 hide_index=True, width="stretch", column_config={
                     "rmse": st.column_config.NumberColumn("RMSE (log)", format="%.3f"),
                     "d_rmse": st.column_config.NumberColumn("Diferença", format="%+.3f"),
                     "d_lo": st.column_config.NumberColumn("IC 95% inf.", format="%+.3f"),
                     "d_hi": st.column_config.NumberColumn("IC 95% sup.", format="%+.3f"),
                     "mdape": st.column_config.NumberColumn("Erro absoluto mediano", format="percent"),
                     "w25": st.column_config.NumberColumn("Dentro de ±25%", format="percent")})

    st.subheader("Estimadores lineares para amostras pequenas")
    st.markdown(
        "Mínimos quadrados ordinários e ponderados, ridge, lasso, elastic net, regressões bayesianas e robustas, PLS, "
        "componentes principais, modelo misto e média de modelos, comparados com o OLS do modelo recomendado. "
        "Penalização, pesos e componentes foram escolhidos só dentro de cada dobra de treino."
    )
    lin = revisao_linear[revisao_linear["esquema"] == esquema].copy()
    lin["Estimador"] = lin["modelo"].map(NOMES_LINEARES).fillna(lin["modelo"])
    lin = lin[lin["modelo"] != "OLS_M5i"].sort_values("d_rmse", ascending=False)
    lin["Resultado"] = np.where(lin["d_hi"] < 0, "Melhor que o OLS", np.where(lin["d_lo"] > 0, "Pior que o OLS", "Sem diferença significativa"))
    lin["erro_mais"] = lin["d_hi"] - lin["d_rmse"]
    lin["erro_menos"] = lin["d_rmse"] - lin["d_lo"]
    base_lin = revisao_linear[(revisao_linear["esquema"] == esquema) & (revisao_linear["modelo"] == "OLS_M5i")]["rmse"].iloc[0]
    fig = px.scatter(lin, x="d_rmse", y="Estimador", color="Resultado", error_x="erro_mais", error_x_minus="erro_menos",
                     color_discrete_map={"Melhor que o OLS": "#1a73e8", "Pior que o OLS": "#d93025", "Sem diferença significativa": "#9aa0a6"},
                     hover_data={"rmse": ":.4f", "d_rmse": ":+.4f"},
                     labels={"d_rmse": f"Diferença de RMSE (log) em relação ao OLS do modelo recomendado (RMSE {br(base_lin, 3)})"},
                     title=f"Estimadores lineares: erro menos o erro do OLS, {esquema}")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=780, legend=dict(orientation="h", y=-0.1), yaxis_title="")
    st.plotly_chart(fig, width="stretch")
    st.info("Com as 5 variáveis do modelo recomendado, todos os estimadores lineares dão praticamente o mesmo erro: "
            "o OLS já está perto do ótimo. A penalização ajuda quando há muitas variáveis (lasso com 14 candidatas), "
            "mas não supera o modelo de 5 variáveis, que o melhor subconjunto por BIC escolhe em todas as dobras.")

    st.subheader("Teste rápido ao vivo")
    st.caption("Roda agora, neste computador, alguns algoritmos com hiperparâmetros fixos e validação LOSO. É uma versão simplificada "
               "do teste acima, para mostrar em aula como a comparação é feita.")
    if st.button("Rodar o teste rápido (cerca de 30 segundos)"):
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        from sklearn.linear_model import LinearRegression

        candidatos = {
            "Regressão com interação (5 var.)": (MODELOS["Recomendado — interação ABL × LOC"], lambda: LinearRegression()),
            "Regressão de referência (5 var.)": (MODELOS["Referência — 5 variáveis"], lambda: LinearRegression()),
            "Regressão com 13 variáveis": (VARS_13, lambda: LinearRegression()),
            "SVR kernel RBF (5 var.)": (MODELOS["Referência — 5 variáveis"], lambda: make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.1))),
            "k-vizinhos (5 var.)": (MODELOS["Referência — 5 variáveis"], lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(7, weights="distance"))),
            "Random forest (13 var.)": (VARS_13, lambda: RandomForestRegressor(300, min_samples_leaf=3, max_features=0.5, random_state=1, n_jobs=-1)),
            "Gradient boosting (13 var.)": (VARS_13, lambda: GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8, random_state=1)),
        }
        y = dados["ln_vu"].values
        grupos = dados["grupo_shopping"].values
        resultados, barra = [], st.progress(0.0)
        for i, (nome, (cols, fabrica)) in enumerate(candidatos.items()):
            X = dados[cols].values
            pred = np.zeros(len(y))
            for g in np.unique(grupos):
                teste = grupos == g
                pred[teste] = fabrica().fit(X[~teste], y[~teste]).predict(X[teste])
            resultados.append({"Algoritmo": nome, "RMSE LOSO (log)": np.sqrt(np.mean((y - pred) ** 2)),
                               "Erro absoluto mediano": np.median(np.abs(np.exp(y - pred) - 1))})
            barra.progress((i + 1) / len(candidatos))
        r = pd.DataFrame(resultados).sort_values("RMSE LOSO (log)")
        st.dataframe(r, hide_index=True, width="stretch", column_config={
            "RMSE LOSO (log)": st.column_config.NumberColumn(format="%.3f"),
            "Erro absoluto mediano": st.column_config.NumberColumn(format="percent")})
        st.success(f"Menor erro: {r.iloc[0]['Algoritmo']}.")

# ----------------------------------------------------------------------------------------------
# 4. Avaliar um shopping
# ----------------------------------------------------------------------------------------------
with aba_avaliar:
    st.markdown("Informe os dados do shopping avaliando. O aplicativo aplica os dois modelos, verifica o domínio e indica qual usar.")
    e1, e2, e3 = st.columns(3)
    UFS = ["AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR",
           "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"]
    uf = e1.selectbox("UF", UFS, index=UFS.index("SP"))
    sugestoes = sorted(cidades.loc[cidades["uf"] == uf, "cidade"])
    cidade = e1.selectbox("Cidade (escolha da lista ou digite qualquer outra)", sugestoes,
                          index=sugestoes.index("Sorocaba") if "Sorocaba" in sugestoes else None,
                          accept_new_options=True, placeholder="Digite o nome da cidade")
    loc_auto, criterio = classificar_loc(cidade or "", uf, cidades)
    with e1.expander("Ajustar a localização manualmente"):
        manual = st.checkbox("Informar LOC manualmente", value=False)
        loc_manual = st.radio("LOC", [0, 1, 2], index=loc_auto, format_func=lambda v: NOMES_LOC[v], disabled=not manual)
    loc = loc_manual if manual else loc_auto
    e1.caption(f"LOC = {loc}: {criterio if not manual else 'informado manualmente'}. Cidades fora das regiões metropolitanas "
               "de SP e RJ recebem LOC 0, mesmo sem transação na amostra.")
    abl = e2.number_input("ABL do shopping (m²)", min_value=5_000, max_value=300_000, value=35_000, step=500)
    lojas = e2.number_input("Número de lojas (mesma data da ABL)", min_value=10, max_value=1_000, value=210, step=1)
    participacao = e3.number_input("Participação avaliada (%)", min_value=0.1, max_value=100.0, value=20.0, step=0.5) / 100
    ano = e3.selectbox("Ano da data-base", list(range(2020, 2027)), index=6)
    vu_arbitrado = e3.number_input("VU arbitrado (opcional, R$/m²)", min_value=0.0, value=0.0, step=100.0,
                                   help="Deixe 0 para adotar a estimativa central. Valor diferente só dentro do campo de arbítrio de ±15%, com justificativa.")
    st.caption("A variável de controle (CTRL) é 1 quando a participação avaliada é de 50% ou mais. Para datas-base após 2026, o modelo deve ser reestimado.")

    dl = lojas / abl * 1000
    entrada = {"ctrl": int(participacao >= 0.5), "ln_abl": np.log(abl), "p2026": int(ano == 2026), "dl": dl, "loc": loc,
               "loc_x_ln_abl": loc * np.log(abl / ABL_REF)}

    # Domínio
    faixa_loc = dados[dados["loc"] == loc]["abl_m2"]
    dl_min, dl_max = dados["dl"].min(), dados["dl"].max()
    dentro_dl = dl_min <= dl <= dl_max
    dentro_abl_loc = faixa_loc.min() <= abl <= faixa_loc.max()
    dentro_abl_geral = dados["abl_m2"].min() <= abl <= dados["abl_m2"].max()

    resultados = {}
    for nome, cols in MODELOS.items():
        m = ajustar_modelo(dados, tuple(cols))
        p = prever(m["res"], cols, entrada)
        p["amplitude"] = (p["ic"][1] - p["ic"][0]) / p["vu"]
        p["grau"] = grau_precisao(p["amplitude"])
        resultados[nome] = p

    if dentro_abl_loc and dentro_dl:
        usar = "Recomendado — interação ABL × LOC"
        st.success(f"Dentro do domínio: use o **modelo recomendado**. ABL entre {br(faixa_loc.min())} e {br(faixa_loc.max())} m² "
                   f"para LOC {loc}; densidade de {br(dl, 2)} lojas por 1.000 m².")
    elif dentro_abl_geral and dentro_dl:
        usar = "Referência — 5 variáveis"
        st.warning(f"A ABL está fora da faixa observada para LOC {loc} ({br(faixa_loc.min())} a {br(faixa_loc.max())} m²): "
                   "o termo de interação extrapolaria. Use o **modelo de referência**.")
    else:
        usar = "Referência — 5 variáveis"
        st.error("ABL ou densidade de lojas fora do domínio da amostra: os dois modelos extrapolam. "
                 f"ABL de {br(dados['abl_m2'].min())} a {br(dados['abl_m2'].max())} m²; densidade de {br(dl_min, 1)} a {br(dl_max, 1)} lojas por 1.000 m².")

    cols_res = st.columns(2)
    for col, (nome, p) in zip(cols_res, resultados.items()):
        with col:
            destaque = " ⭐" if nome == usar else ""
            st.subheader(nome + destaque)
            a, b_ = st.columns(2)
            a.metric("VU central (R$/m²)", br(p["vu"]))
            b_.metric("Valor da participação", f"R$ {br(p['vu'] * abl * participacao / 1e6, 1)} mi")
            tabela = pd.DataFrame({
                "Grandeza": ["VU, intervalo de confiança de 80% (R$/m²)", "VU, intervalo de predição de 80% (R$/m²)",
                             "Campo de arbítrio ±15% (R$/m²)", "Valor de 100% do shopping (R$ mi)", "Valor da participação (R$ mi)"],
                "Inferior": [br(p["ic"][0]), br(p["ip"][0]), br(p["vu"] * 0.85), br(p["ic"][0] * abl / 1e6, 1), br(p["ic"][0] * abl * participacao / 1e6, 1)],
                "Central": [br(p["vu"]), br(p["vu"]), br(p["vu"]), br(p["vu"] * abl / 1e6, 1), br(p["vu"] * abl * participacao / 1e6, 1)],
                "Superior": [br(p["ic"][1]), br(p["ip"][1]), br(p["vu"] * 1.15), br(p["ic"][1] * abl / 1e6, 1), br(p["ic"][1] * abl * participacao / 1e6, 1)],
            })
            st.dataframe(tabela, hide_index=True, width="stretch")
            st.markdown(f"Amplitude do IC 80%: **{pct(p['amplitude'], 1)}** · grau de precisão **{p['grau']}** (NBR 14653-2)")

    st.subheader("Valor adotado")
    p_usar = resultados[usar]
    arb_inf, arb_sup = p_usar["vu"] * 0.85, p_usar["vu"] * 1.15
    vu_adotado = vu_arbitrado if vu_arbitrado > 0 else p_usar["vu"]
    v1, v2, v3 = st.columns(3)
    v1.metric("VU adotado (R$/m²)", br(vu_adotado))
    v2.metric("Valor de 100% do shopping", f"R$ {br(vu_adotado * abl / 1e6, 1)} mi")
    v3.metric("Valor da participação avaliada", f"R$ {br(vu_adotado * abl * participacao / 1e6, 1)} mi")
    if vu_arbitrado <= 0:
        st.info(f"Adotada a estimativa central do modelo indicado ({usar.split(' —')[0].lower()}). "
                f"Campo de arbítrio: {br(arb_inf)} a {br(arb_sup)} R$/m².")
    elif arb_inf <= vu_arbitrado <= arb_sup:
        dentro_ic = p_usar["ic"][0] <= vu_arbitrado <= p_usar["ic"][1]
        st.warning(f"VU arbitrado dentro do campo de arbítrio ({br(arb_inf)} a {br(arb_sup)} R$/m²), "
                   f"{'dentro' if dentro_ic else 'fora'} do intervalo de confiança de 80%. "
                   "A NBR 14653-2 exige justificar no laudo as características do imóvel não contempladas no modelo.")
    else:
        st.error(f"VU arbitrado fora do campo de arbítrio de ±15% ({br(arb_inf)} a {br(arb_sup)} R$/m²): não admitido.")

    with st.expander("Memória de cálculo"):
        for nome, cols in MODELOS.items():
            m = ajustar_modelo(dados, tuple(cols))
            linhas = [f"- {ROTULO_CURTO[c]} = {br(entrada[c], 4)} × {br(m['res'].params[c], 4)} = {br(entrada[c] * m['res'].params[c], 4)}" for c in cols]
            soma = m["res"].params["const"] + sum(entrada[c] * m["res"].params[c] for c in cols)
            st.markdown(f"**{nome}**\n\n- Constante = {br(m['res'].params['const'], 4)}\n" + "\n".join(linhas) +
                        f"\n\nln(VU) = {br(soma, 4)} → VU = exp({br(soma, 4)}) = **{br(np.exp(soma))} R\\$/m²**")
        st.caption(f"DL = {lojas} lojas ÷ {br(abl)} m² × 1.000 = {br(dl, 3)}. CTRL = {entrada['ctrl']}. P2026 = {entrada['p2026']}.")

    fig = px.scatter(dados, x="abl_m2", y="vu_rs_m2", color="localizacao", log_x=True, log_y=True, opacity=0.6,
                     color_discrete_map={v: CORES_LOC[k] for k, v in NOMES_LOC.items()}, hover_data=["shopping", "ano"],
                     labels={"abl_m2": "ABL (m²)", "vu_rs_m2": "VU (R$/m²)", "localizacao": ""},
                     title="O avaliando entre as transações da amostra (VU observado, sem ajuste)")
    p = resultados[usar]
    fig.add_trace(go.Scatter(x=[abl], y=[p["vu"]], mode="markers", name="Avaliando",
                             marker=dict(symbol="star", size=20, color="#f9ab00", line=dict(color="black", width=1)),
                             error_y=dict(type="data", array=[p["ip"][1] - p["vu"]], arrayminus=[p["vu"] - p["ip"][0]], color="#f9ab00")))
    fig.update_layout(legend=dict(orientation="h", y=-0.2), height=480)
    st.plotly_chart(fig, width="stretch")
    st.caption("A barra do avaliando é o intervalo de predição de 80%: a faixa em que se espera o preço de uma transação real.")

# ----------------------------------------------------------------------------------------------
# 5. Método e conclusões
# ----------------------------------------------------------------------------------------------
with aba_metodo:
    st.markdown("""
### Como o modelo foi construído

1. **Amostra.** 176 registros de transações (planilha de mercado 2020–2024, arquivo de transações 2023–2026 e 14 negócios levantados na web). Ficaram 80 transações de um único shopping com valor, participação, ABL e número de lojas.
2. **Variável dependente.** ln(VU), com VU = valor ÷ (participação × ABL), em R\\$/m² de ABL referido a 100%.
3. **Busca exaustiva.** 14 variáveis candidatas combinadas em 5.498 regressões, julgadas pelo erro de validação cruzada e pelo BIC.
4. **Testes de refutação.** Tendência anual linear (refutada no teste de 2026), percentual contínuo (não significativo), variáveis do censo (não melhoram a previsão).
5. **Comparação com 30 algoritmos** em validação cruzada aninhada. Árvores, boosting e rede neural erram de 12% a 25% mais. Os modelos flexíveis que ganham revelam a interação entre porte e localização.
6. **Modelo recomendado.** A interação escrita na regressão: cinco variáveis, erro 11% menor para shopping novo, único ganho estatisticamente significativo.
7. **Revisão da modelagem linear.** Mínimos quadrados ponderados, ridge, lasso, elastic net, regressões bayesianas e robustas, PLS, componentes principais e média de modelos dão o mesmo erro do OLS com as 5 variáveis, ou maior. O OLS continua sendo o estimador adotado.
8. **Localização de qualquer cidade.** LOC = 2 na capital paulista; 1 no Rio de Janeiro capital e nos demais municípios das regiões metropolitanas de São Paulo (39) e do Rio de Janeiro (22); 0 em qualquer outra cidade do país.

### Principais achados

- Localização é o fator dominante e age junto com o porte: o porte só eleva o valor por m² nas metrópoles de SP e RJ.
- Cada loja adicional por 1.000 m² de ABL eleva o VU em cerca de 23%.
- Blocos de controle saem cerca de 42% abaixo das frações, por m², sobretudo por composição: os ativos premium só negociaram frações.
- Preços nominais estáveis de 2020 a 2025 e novo patamar em 2026 (+36%).
- Com 80 observações, a estrutura certa vale mais que a flexibilidade do algoritmo.

### Limites

- Use o modelo recomendado só dentro da faixa de ABL de cada localização; fora dela, o de referência.
- O valor adotado é a estimativa central; outro valor só dentro do campo de arbítrio de ±15%, com justificativa.
- O intervalo de confiança analítico supõe transações independentes; o bootstrap por shopping dá intervalos cerca de 4 pontos percentuais mais largos.
- Lojas e perfil de público vêm do Censo ABRASCE 2016/2017.
- Os erros de validação são algo otimistas, porque as variáveis foram escolhidas sobre os mesmos dados.
- Para datas-base após 2026, reestime o modelo com as novas transações.
""")
    st.caption("Relatório completo: Relatório — Valor de shopping centers por transações 2020–2026. Planilha: Modelo shopping centers 2020-2026.xlsx.")
