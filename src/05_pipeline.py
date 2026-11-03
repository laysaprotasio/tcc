"""Etapas 5 e 6: protocolo do artigo (CL) + avaliacoes adicionais
(CV espacial, teste sem poda, varias sementes)."""
import argparse, json, warnings
import numpy as np, pandas as pd
from cleanlab.filter import find_label_issues
from cleanlab.count import estimate_joint
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score, average_precision_score,
                             balanced_accuracy_score)
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import *
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--smote-ratio", type=float, default=1 / 3,
                help="positivo/negativo apos SMOTE (1:3 -> 1/3 -> 25 por cento positivos)")
ap.add_argument("--bloco-graus", type=float, default=0.05, help="lado do bloco espacial para o GroupKFold")
ap.add_argument("--corr", type=float, default=0.80)
a = ap.parse_args()

ds = pd.read_csv(DATA / "dataset_rotulado.csv.gz")
y_all = ds.y.values.astype(int)


def remover_correlacionadas(X, lim):
    """Remove iterativamente a variavel com mais pares |r|>=lim (desempate: maior |r| medio)."""
    corr = X.corr().abs()
    cols, drop = list(X.columns), []
    while True:
        c = corr.loc[cols, cols].copy()
        c = pd.DataFrame(c.to_numpy(copy=True), index=cols, columns=cols)
        c = c.mask(np.eye(len(cols), dtype=bool), 0.0)
        viol = (c >= lim).sum()
        if viol.max() == 0:
            break
        cand = viol[viol == viol.max()].index
        d = c.loc[cand].mean(axis=1).idxmax()
        drop.append(d); cols.remove(d)
    return cols, drop


cols, removidas = remover_correlacionadas(ds[PREDITORAS], a.corr)
X_all = ds[cols].values
ds[PREDITORAS].corr().round(3).to_csv(RES / "correlacao_pearson.csv")
print(f"Removidas por |r|>={a.corr}: {removidas} -> restam {cols} (artigo removeu 3)")


def modelo():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def smote(X, y, seed):
    k = max(1, min(5, int(y.sum()) - 1))
    return SMOTE(sampling_strategy=a.smote_ratio, k_neighbors=k, random_state=seed).fit_resample(X, y)


def cl(Xs, ys, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pp = cross_val_predict(modelo(), Xs, ys, cv=cv, method="predict_proba")
    issues = find_label_issues(labels=ys.copy(), pred_probs=pp, filter_by="prune_by_noise_rate")
    t = [float(pp[ys == j, j].mean()) for j in (0, 1)]          # auto-confianca t_j (Eq. 3 do artigo)
    Q = estimate_joint(labels=ys, pred_probs=pp)               # linhas = rotulo ruidoso, colunas = verdadeiro
    return issues, t, Q


def cv_art(X, y, seed):
    r = cross_validate(modelo(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                       scoring=["accuracy", "recall", "roc_auc"])
    return {"acc": r["test_accuracy"].mean(), "recall": r["test_recall"].mean(), "auc": r["test_roc_auc"].mean()}


def avalia(m, X, y):
    p = m.predict_proba(X)[:, 1]
    yh = (p >= 0.5).astype(int)
    return {"acc": accuracy_score(y, yh), "recall": recall_score(y, yh), "auc": roc_auc_score(y, p),
            "ap": average_precision_score(y, p), "bal_acc": balanced_accuracy_score(y, yh)}


def tres_bracos(Xtr, ytr, Xte, yte, seed):
    """baseline ruidoso | SMOTE | SMOTE+CL; todos testados nos rotulos ORIGINAIS, sem poda e sem SMOTE."""
    out = {"baseline": avalia(modelo().fit(Xtr, ytr), Xte, yte)}
    Xs, ys = smote(Xtr, ytr, seed)
    out["smote"] = avalia(modelo().fit(Xs, ys), Xte, yte)
    iss, _, _ = cl(Xs, ys, seed)
    out["smote_cl"] = avalia(modelo().fit(Xs[~iss], ys[~iss]), Xte, yte)
    out["smote_cl"]["frac_podada_treino"] = float(iss.mean())
    return out


bl = (np.floor(ds.Latitude / a.bloco_graus).astype(int) * 100000 + np.floor(ds.Longitude / a.bloco_graus).astype(int))
grupos = pd.factorize(bl)[0]

linhas, detalhe_cl = [], []
for s in range(a.seeds):
    seed = 42 + s
    # (a) protocolo do artigo: SMOTE no conjunto inteiro, CL, CV aleatoria no conjunto podado
    Xs, ys = smote(X_all, y_all, seed)
    iss, t, Q = cl(Xs, ys, seed)
    detalhe_cl.append({"seed": seed, "n_smote": len(ys), "prop_pos_smote": float(ys.mean()), "t0": t[0], "t1": t[1],
                       "Q00": Q[0, 0], "Q01": Q[0, 1], "Q10": Q[1, 0], "Q11": Q[1, 1],
                       "frac_suspeitos_prune_by_noise_rate": float(iss.mean()), "frac_offdiag_Q": float(Q[0, 1] + Q[1, 0])})
    for nome, (X, y) in {"A_original(SMOTE,CV aleat.)": (Xs, ys), "A_CL(podado,CV aleat.)": (Xs[~iss], ys[~iss])}.items():
        linhas.append({"seed": seed, "protocolo": nome, **cv_art(X, y, seed), "n": len(y)})
    # (b1) CV espacial: SMOTE(+CL) so no treino; teste com rotulos originais
    folds = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X_all, y_all, grupos):
        if y_all[te].sum() == 0 or y_all[tr].sum() < 6:
            continue
        folds.append(tres_bracos(X_all[tr], y_all[tr], X_all[te], y_all[te], seed))
    for braco in ("baseline", "smote", "smote_cl"):
        m = pd.DataFrame([f[braco] for f in folds]).mean().to_dict() if folds else {}
        linhas.append({"seed": seed, "protocolo": f"B1_CV_espacial_teste_sem_poda/{braco}", "n_folds_validos": len(folds), **m})
    # (b2) holdout aleatorio 25%: treino SMOTE(+CL), teste sem poda
    tr, te = train_test_split(np.arange(len(y_all)), test_size=0.25, stratify=y_all, random_state=seed)
    for braco, m in tres_bracos(X_all[tr], y_all[tr], X_all[te], y_all[te], seed).items():
        linhas.append({"seed": seed, "protocolo": f"B2_holdout_aleatorio_sem_poda/{braco}", **m})
    print(f"semente {seed} ok", flush=True)

res = pd.DataFrame(linhas)
res.to_csv(RES / "metricas_por_semente.csv", index=False)
cld = pd.DataFrame(detalhe_cl)
cld.to_csv(RES / "cl_detalhe_por_semente.csv", index=False)
metr = [c for c in ["acc", "recall", "auc", "ap", "bal_acc", "frac_podada_treino", "n_folds_validos"] if c in res]
resumo = res.groupby("protocolo")[metr].agg(["mean", "std"]).round(4)
resumo.to_csv(RES / "metricas_resumo_media_dp.csv")
alvo = {"original": {"acc": .767, "recall": .50, "auc": .781}, "CL": {"acc": .956, "recall": .75, "auc": .958},
        "t0": .799, "t1": .331, "Q": [[.624, .169], [.073, .134]], "frac_incorretos": .242}
(RES / "resumo.json").write_text(json.dumps({
    "variaveis_usadas": cols, "removidas_correlacao": removidas, "n_linhas": len(ds), "n_pos": int(y_all.sum()),
    "cl_media_dp": cld.drop(columns="seed").agg(["mean", "std"]).round(4).to_dict(), "artigo": alvo}, indent=2))
pd.set_option("display.width", 220)
print(resumo.to_string())
print(cld.drop(columns="seed").agg(["mean", "std"]).round(4).T.to_string())
