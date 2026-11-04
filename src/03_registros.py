"""Etapa 2 (cont.): filtra janelas dos voos, tenta SiBBr, deduplica e compara com os 54 do artigo."""
import argparse, json, sys
import numpy as np, pandas as pd, requests
from common import *

LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = -3.632, -2.813, -60.831, -59.937
ap = argparse.ArgumentParser()
ap.add_argument("--criterio", choices=["janelas", "ano2014"], default="janelas",
                help="janelas: so datas dos voos; ano2014: 2014 inteiro, IOP = janela mais proxima no tempo")
ap.add_argument("--ignorar-contagem", action="store_true")
a = ap.parse_args()
passos = {"criterio": a.criterio}
g = pd.read_csv(DATA / "gbif_raw.csv"); passos["gbif_bruto"] = len(g)
g["data"] = pd.to_datetime(g["eventDate"].astype(str).str.slice(0, 10), errors="coerce")
passos["gbif_sem_data_valida"] = int(g["data"].isna().sum())
if "occurrenceStatus" in g:
    g = g[g["occurrenceStatus"].fillna("PRESENT").str.upper() != "ABSENT"]
passos["gbif_apos_remover_ABSENT"] = len(g)
g = g.dropna(subset=["data"])
g["IOP"] = g["data"].map(iop_da_data if a.criterio == "janelas" else iop_mais_proxima)
if a.criterio == "ano2014":
    g = g[g["data"].dt.year == 2014]
passos["gbif_na_janela_IOP1"] = int((g.IOP == 1).sum()); passos["gbif_na_janela_IOP2"] = int((g.IOP == 2).sum())
g["fonte"] = "GBIF"
if "coordinateUncertaintyInMeters" in g:
    passos["gbif_incerteza_coord_mediana_m"] = float(g.coordinateUncertaintyInMeters.median()) if g.coordinateUncertaintyInMeters.notna().any() else None
g = g[g.IOP > 0].rename(columns={"decimalLatitude": "lat", "decimalLongitude": "lon"})
g = g[["lat", "lon", "data", "IOP", "fonte", "datasetName"] if "datasetName" in g else ["lat","lon","data","IOP","fonte"]]

# --- SiBBr (no brasileiro do GBIF; agrega ICMBio) ---
sib = pd.DataFrame(); sib_status = "nao_tentado"
for url in ["https://biocache.sibbr.gov.br/ws/occurrences/search", "https://biocache-ws.sibbr.gov.br/occurrences/search"]:
    try:
        r = requests.get(url, params={"q": 'taxon_name:"Coragyps atratus"', "pageSize": 1000,
              "fq": ["year:[2014 TO 2015]", f"latitude:[{LAT_MIN} TO {LAT_MAX}]", f"longitude:[{LON_MIN} TO {LON_MAX}]"]}, timeout=30)
        r.raise_for_status()
        occs = r.json().get("occurrences", [])
        sib = pd.DataFrame([{"lat": o.get("decimalLatitude"), "lon": o.get("decimalLongitude"),
                             "data": pd.to_datetime(o.get("eventDate"), unit="ms", errors="coerce")} for o in occs])
        sib_status = f"ok ({url}): {len(sib)} registros"; break
    except Exception as e:
        sib_status = f"falhou ({type(e).__name__}: {str(e)[:100]})"
passos["sibbr"] = sib_status
if len(sib):
    sib["IOP"] = sib["data"].map(iop_da_data if a.criterio == "janelas" else iop_mais_proxima); sib = sib[sib.IOP > 0]; sib["fonte"] = "SiBBr"
    g = pd.concat([g, sib], ignore_index=True)
passos["antes_dedup"] = len(g)
g["_k"] = g.lat.round(4).astype(str) + "|" + g.lon.round(4).astype(str) + "|" + g.data.dt.date.astype(str)
g = g.drop_duplicates("_k").drop(columns="_k")
# tambem dentro do retangulo exato do CSV
g = g[g.lat.between(LAT_MIN, LAT_MAX) & g.lon.between(LON_MIN, LON_MAX)]
passos["apos_dedup_e_bbox"] = len(g)
g.to_csv(DATA / "presencas.csv", index=False)
g.to_csv(DATA / f"presencas_{a.criterio}.csv", index=False)
n = len(g)
passos["artigo_n_registros"] = ARTIGO_N_REGISTROS
passos["diferenca_vs_artigo"] = n - ARTIGO_N_REGISTROS
passos["razao_vs_artigo"] = round(n / ARTIGO_N_REGISTROS, 2)
passos["muito_diferente"] = bool(n < 0.5 * ARTIGO_N_REGISTROS or n > 1.5 * ARTIGO_N_REGISTROS)
(RES / f"registros_etapas_{a.criterio}.json").write_text(json.dumps(passos, indent=2, ensure_ascii=False, default=str))
print(json.dumps(passos, indent=2, ensure_ascii=False, default=str))
if passos["muito_diferente"] and not a.ignorar_contagem:
    print("\nPARANDO: total de registros muito diferente dos 54 do artigo. Reveja com a orientadora/rode com --ignorar-contagem.")
    sys.exit(3)
