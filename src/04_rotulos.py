"""Etapa 4: monta o conjunto rotulado (juncao presenca x grade) e a tabela raio x proporcao."""
import argparse, json
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--iops", default="1,2", help="IOPs incluidos no conjunto (ex.: 1,2 | 1 | 2)")
ap.add_argument("--raio-m", type=float, default=None,
                help="raio do buffer (m); padrao: o que deixa a proporcao mais perto de 3,3 por cento")
ap.add_argument("--max-match-m", type=float, default=5000, help="distancia maxima registro->celula mais proxima (m)")
ap.add_argument("--raios", default="0,100,250,500,1000,2000,3000,5000,10000")
a = ap.parse_args()

g = carregar_grade()
rec = pd.read_csv(DATA / "presencas.csv", parse_dates=["data"])
lat0 = np.deg2rad(g.Latitude.mean())
def proj(lat, lon):  # metros (aproximacao local equiretangular)
    return np.c_[np.asarray(lon) * np.cos(lat0) * 111_320.0, np.asarray(lat) * 110_570.0]
g["complete"] = g[PREDITORAS].notna().all(axis=1)

trees = {}
for iop in (1, 2):
    sub = g[(g.IOP == iop) & g.complete]
    trees[iop] = (cKDTree(proj(sub.Latitude.values, sub.Longitude.values)), sub.index.values)

dists, cels = [], []
for _, r in rec.iterrows():
    tree, idx = trees[int(r.IOP)]
    d, j = tree.query(proj([r.lat], [r.lon]))
    dists.append(float(d[0])); cels.append(int(idx[j[0]]))
rec["dist_celula_m"] = dists; rec["celula_idx"] = cels
rec["casou"] = rec.dist_celula_m <= a.max_match_m
rec.to_csv(RES / "registros_juncao.csv", index=False)
dist = rec.dist_celula_m.describe().round(1).to_dict()

def positivos(raio, iops):
    """Mascara (len(g)) das celulas positivas: celula mais proxima de cada registro + celulas da mesma IOP dentro do raio."""
    pos = np.zeros(len(g), bool)
    for _, r in rec[rec.casou & rec.IOP.isin(iops)].iterrows():
        pos[int(r.celula_idx)] = True
        if raio > 0:
            tree, idx = trees[int(r.IOP)]
            for j in tree.query_ball_point(proj([r.lat], [r.lon]), raio)[0]:
                pos[idx[j]] = True
    return pos

raios = [float(x) for x in a.raios.split(",")]
tabela = []
for opc in ([1, 2], [1], [2]):
    em = (g.complete & g.IOP.isin(opc)).values
    n_lin = int(em.sum())
    for r in raios:
        npos = int(positivos(r, opc)[em].sum())
        tabela.append({"iops": "+".join(map(str, opc)), "raio_m": r, "n_linhas": n_lin, "n_pos": npos,
                       "prop_pos": npos / n_lin, "dif_vs_3.3pct": npos / n_lin - ALVO_N_PROP})
tab = pd.DataFrame(tabela)
tab.to_csv(RES / "tabela_raio_vs_positivos.csv", index=False)
print(tab.round(4).to_string(index=False))

iops = [int(x) for x in a.iops.split(",")]
cand = tab[tab.iops == "+".join(map(str, iops))]
raio = a.raio_m if a.raio_m is not None else float(cand.loc[cand["dif_vs_3.3pct"].abs().idxmin(), "raio_m"])
y = positivos(raio, iops)
ds = g[g.complete & g.IOP.isin(iops)].copy()
ds["y"] = y[ds.index.values].astype(int)
ds.drop(columns=["complete"]).to_csv(DATA / "dataset_rotulado.csv.gz", index=False)
meta = {"iops": iops, "raio_m": raio, "max_match_m": a.max_match_m, "n_linhas": len(ds), "n_pos": int(ds.y.sum()),
        "prop_pos": float(ds.y.mean()), "artigo": {"n_linhas": ARTIGO_N_LINHAS, "prop_pos": ALVO_N_PROP},
        "n_registros_usados": int(rec[rec.casou & rec.IOP.isin(iops)].shape[0]),
        "n_registros_nao_casaram": int((~rec.casou).sum()), "dist_registro_celula_m": dist}
(RES / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
