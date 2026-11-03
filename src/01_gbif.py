"""Etapa 2: busca de registros de Coragyps atratus no GBIF (e tentativa no SiBBr)."""
import datetime, json, sys, time
from pathlib import Path
import pandas as pd
from pygbif import occurrences as occ, species

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
DATA.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)

LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = -3.632, -2.813, -60.831, -59.937
# WKT anti-horario: (minLon,minLat)->(maxLon,minLat)->(maxLon,maxLat)->(minLon,maxLat)
WKT = (f"POLYGON(({LON_MIN} {LAT_MIN},{LON_MAX} {LAT_MIN},{LON_MAX} {LAT_MAX},"
       f"{LON_MIN} {LAT_MAX},{LON_MIN} {LAT_MIN}))")
YEARS = "2014,2015"
SPECIES = "Coragyps atratus"

def busca(params, offset, tentativas=12):
    """Busca com retentativas (a API do GBIF devolve 503 de forma intermitente)."""
    for i in range(tentativas):
        try:
            return occ.search(**params, limit=300, offset=offset)
        except Exception as e:  # noqa
            espera = min(30 * (i + 1), 300)
            print(f"  falha ({type(e).__name__}: {str(e)[:80]}); retentando em {espera}s", flush=True)
            time.sleep(espera)
    raise RuntimeError("GBIF indisponivel apos varias tentativas")

def main():
    bb = species.name_backbone(scientificName=SPECIES)
    key = bb["usage"]["key"] if "usage" in bb else bb["usageKey"]
    params = dict(scientificName=SPECIES, geometry=WKT, hasCoordinate=True, year=YEARS)
    rows, offset, total = [], 0, None
    while True:
        r = busca(params, offset)
        total = r["count"]
        rows += r["results"]
        offset += 300
        if r["endOfRecords"] or not r["results"]:
            break
    df = pd.json_normalize(rows)
    keep = [c for c in ["key", "datasetKey", "publishingOrgKey", "institutionCode", "collectionCode",
            "basisOfRecord", "decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters",
            "eventDate", "year", "month", "day", "individualCount", "occurrenceStatus",
            "scientificName", "datasetName", "license"] if c in df.columns]
    df = df[keep]
    df.to_csv(DATA / "gbif_raw.csv", index=False)
    log = {"data_consulta": datetime.datetime.now().isoformat(timespec="seconds"),
           "fonte": "GBIF API via pygbif.occurrences.search", "taxonKey_backbone": key, "parametros": {**params, "year": YEARS},
           "wkt": WKT, "n_total_api": total, "n_baixado": len(df)}
    if "occurrenceStatus" in df: log["occurrenceStatus"] = df["occurrenceStatus"].value_counts().to_dict()
    if "basisOfRecord" in df: log["basisOfRecord"] = df["basisOfRecord"].value_counts().to_dict()
    if "year" in df: log["por_ano"] = df["year"].value_counts().to_dict()
    if "datasetName" in df: log["por_dataset"] = df["datasetName"].fillna("?").value_counts().to_dict()
    (RES / "gbif_consulta.json").write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(log, indent=2, ensure_ascii=False, default=str))

if __name__ == "__main__":
    main()
