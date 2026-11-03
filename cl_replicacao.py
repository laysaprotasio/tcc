"""
Replicação aproximada do pipeline de Confident Learning (CL) para
Modelagem de Distribuição de Espécies (SDM), conforme descrito em
Miyaji, Almeida e Corrêa (SBBD 2023), seções 3 e 4.

Protocolo do artigo (reproduzido aqui):
  1. Remover uma variável de cada par com |Pearson| >= 0.80 entre preditoras
  2. SMOTE ate proporcao 1:3 (positivo:negativo)  -> ~23% de positivos
  3. Regressao Logistica (C=1), validacao cruzada 5-fold estratificada
     para obter probabilidades fora da amostra
  4. CL (Northcutt et al. 2021): limiares t_j, matriz C, Q, poda
  5. Retreina e avalia com 5-fold estratificado (acuracia, revocacao, ROC-AUC)

ATENCAO:
  * O dataset original (urubu-preto + meteorologia/aerossois, regiao de Manaus)
    nao acompanha o artigo. Aqui usamos dados SINTETICOS com ruido injetado.
    Para dados reais, substitua `carregar_dados()` (X DataFrame, y 0/1).
  * No artigo, as metricas "depois do CL" sao medidas por CV no proprio
    conjunto ja podado. Isso favorece o resultado (os exemplos dificeis foram
    removidos tambem da avaliacao). Por isso o script mostra DUAS avaliacoes:
      (a) protocolo do artigo; e
      (b) avaliacao em teste separado com rotulos verdadeiros (so possivel
          porque os dados sao sinteticos).

Instalacao:  pip install cleanlab scikit-learn imbalanced-learn pandas numpy
"""
import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
RNG = np.random.default_rng(SEED)


def carregar_dados(frac_pos_vira_neg=0.15, frac_neg_vira_pos=0.0):
    """Sinteticos e desbalanceados (~10% positivos). Em SDM a 'ausencia' e
    pouco confiavel (presence-only): parte das ausencias sao presencas nao
    registradas. Devolve X, y observado (ruidoso) e y verdadeiro."""
    X, y_true = make_classification(
        n_samples=6000, n_features=20, n_informative=8, n_redundant=4,
        weights=[0.9, 0.1], class_sep=1.0, flip_y=0.0, random_state=SEED)
    y_obs = y_true.copy()
    pos = np.where(y_true == 1)[0]
    y_obs[RNG.choice(pos, int(frac_pos_vira_neg * len(pos)), replace=False)] = 0
    neg = np.where(y_true == 0)[0]
    y_obs[RNG.choice(neg, int(frac_neg_vira_pos * len(neg)), replace=False)] = 1
    cols = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), y_obs, y_true


def remover_correlacionadas(X, limiar=0.80):
    """Remove uma variavel de cada par com |Pearson| >= limiar (como no artigo)."""
    corr = X.corr().abs()
    drop = set()
    cols = list(X.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a not in drop and b not in drop and corr.loc[a, b] >= limiar:
                drop.add(b)
    return [c for c in cols if c not in drop], sorted(drop)


def modelo():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def cv_metricas(X, y, nome):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    r = cross_validate(modelo(), X, y, cv=cv,
                       scoring=["accuracy", "recall", "roc_auc"])
    print(f"  {nome:<26} Acc={r['test_accuracy'].mean():.3f}  "
          f"Recall={r['test_recall'].mean():.3f}  "
          f"ROC-AUC={r['test_roc_auc'].mean():.3f}  (n={len(y)})")


def teste_limpo(Xtr, ytr, Xte, yte_true, nome):
    m = modelo().fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    yhat = m.predict(Xte)
    print(f"  {nome:<26} Acc={accuracy_score(yte_true, yhat):.3f}  "
          f"Recall={recall_score(yte_true, yhat):.3f}  "
          f"ROC-AUC={roc_auc_score(yte_true, p):.3f}  (n_treino={len(ytr)})")


def main():
    X, y_obs, y_true = carregar_dados()

    # separa um teste com rotulos VERDADEIROS (so existe porque e sintetico)
    X_tr, X_te, yobs_tr, _, ytrue_tr, yte_true = train_test_split(
        X, y_obs, y_true, test_size=0.25, stratify=y_true, random_state=SEED)

    # 1. correlacao de Pearson entre preditoras
    cols, removidas = remover_correlacionadas(X_tr)
    X_tr, X_te = X_tr[cols], X_te[cols]
    print(f"Preditoras removidas por |r|>=0.80: {removidas}  (restam {len(cols)})")

    # 2. SMOTE ate 1:3 (positivo:negativo)
    sm = SMOTE(sampling_strategy=1 / 3, random_state=SEED)
    Xs, ys = sm.fit_resample(X_tr, yobs_tr)
    # nota: amostras sinteticas nao tem rotulo "verdadeiro"; marcamos como
    # positivas verdadeiras (sao interpoladas de positivos observados)
    n_orig = len(yobs_tr)
    ytrue_s = np.concatenate([ytrue_tr, np.ones(len(ys) - n_orig, dtype=int)])
    print(f"Apos SMOTE: {len(ys)} linhas, {ys.mean():.1%} positivas")

    # 3. probabilidades fora da amostra (5-fold estratificado)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred_probs = cross_val_predict(modelo(), Xs, ys, cv=cv, method="predict_proba")

    # 4. CL: poda por taxa de ruido (variante do artigo original de CL)
    issues = find_label_issues(labels=ys.copy(), pred_probs=pred_probs,
                               filter_by="prune_by_noise_rate")
    print(f"CL sinalizou {issues.sum()} de {len(ys)} ({issues.mean():.1%}) rotulos")
    prec = (ys[issues] != ytrue_s[issues]).mean()
    rec = issues[ys != ytrue_s].mean()
    print(f"  precisao da deteccao={prec:.2f}  cobertura do ruido real={rec:.2f}")

    Xc, yc = Xs[~issues], ys[~issues]

    # 5a. protocolo do artigo: CV no proprio conjunto
    print("\n(a) Protocolo do artigo - CV 5-fold no proprio conjunto:")
    cv_metricas(Xs, ys, "SMOTE (original)")
    cv_metricas(Xc, yc, "SMOTE + CL (podado)")

    # 5b. avaliacao honesta: teste separado com rotulos verdadeiros
    print("\n(b) Teste separado com rotulos verdadeiros:")
    teste_limpo(X_tr, yobs_tr, X_te, yte_true, "Baseline ruidoso")
    teste_limpo(Xs, ys, X_te, yte_true, "SMOTE")
    teste_limpo(Xc, yc, X_te, yte_true, "SMOTE + CL")


if __name__ == "__main__":
    main()
