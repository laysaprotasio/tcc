"""Etapa 3: perfil do CSV de preditores."""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "goamazon-climatic-data-202107.csv")
df.columns = [c.split(" (")[0].strip() for c in df.columns]
gas = [c for c in df.columns if c not in ("Latitude", "Longitude", "IOP")]
out = {"n_linhas": len(df), "colunas": list(df.columns)}
out["faltantes"] = {c: {"n": int(df[c].isna().sum()), "pct": round(100 * df[c].isna().mean(), 2)} for c in df.columns}
out["linhas_completas"] = int(df.dropna().shape[0])
out["linhas_completas_por_IOP"] = df.dropna().groupby("IOP").size().to_dict()
out["linhas_por_IOP"] = df.groupby("IOP").size().to_dict()
out["bbox"] = {"lat": [df.Latitude.min(), df.Latitude.max()], "lon": [df.Longitude.min(), df.Longitude.max()]}
# resolucao da grade
la, lo = np.sort(df.Latitude.unique()), np.sort(df.Longitude.unique())
out["n_lat_unicas"], out["n_lon_unicas"] = len(la), len(lo)
out["passo_lat_mediano"] = float(np.median(np.diff(la))); out["passo_lon_mediano"] = float(np.median(np.diff(lo)))
out["celulas_unicas_latlon"] = int(df[["Latitude", "Longitude"]].drop_duplicates().shape[0])
out["celulas_unicas_por_IOP"] = df.groupby("IOP").apply(lambda g: len(g[["Latitude", "Longitude"]].drop_duplicates())).to_dict()
out["linhas_duplicadas_latlon_IOP"] = int(df.duplicated(["Latitude", "Longitude", "IOP"]).sum())
(ROOT / "results").mkdir(exist_ok=True)
df.groupby("IOP")[gas].describe().T.to_csv(ROOT / "results/perfil_estatisticas_por_IOP.csv")
(ROOT / "results/perfil_csv.json").write_text(json.dumps(out, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(df.groupby("IOP")[gas].agg(["mean", "std"]).round(3).T)
print(df[gas].corr().round(2))
