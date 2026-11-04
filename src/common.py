"""Constantes e utilitarios compartilhados."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
DATA.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
CSV = ROOT / "goamazon-climatic-data-202107.csv"
PREDITORAS = ["CO", "O3", "NOX", "CPC3010", "Acetonitrile", "Isoprene", "CH4", "Temperature"]
ALVO_N_PROP = 0.033          # proporcao de positivos do artigo
ARTIGO_N_REGISTROS = 54
ARTIGO_N_LINHAS = 185_305

def carregar_grade():
    df = pd.read_csv(CSV)
    df.columns = [c.split(" (")[0].strip() for c in df.columns]
    return df

def iop_da_data(d):
    """IOP1: 1 fev-31 mar 2014; IOP2: 15 ago-15 out 2014. Fora disso: 0."""
    if pd.isna(d): return 0
    if pd.Timestamp("2014-02-01") <= d <= pd.Timestamp("2014-03-31 23:59:59"): return 1
    if pd.Timestamp("2014-08-15") <= d <= pd.Timestamp("2014-10-15 23:59:59"): return 2
    return 0


def iop_mais_proxima(d):
    """Para o criterio 'ano2014': registro dentro de uma janela -> sua IOP; fora -> IOP cuja janela esta
    temporalmente mais proxima (em dias). SUPOSICAO nossa, nao descrita no artigo."""
    i = iop_da_data(d)
    if i or pd.isna(d): return i
    j1 = (pd.Timestamp("2014-02-01"), pd.Timestamp("2014-03-31"))
    j2 = (pd.Timestamp("2014-08-15"), pd.Timestamp("2014-10-15"))
    dist = lambda j: 0 if j[0] <= d <= j[1] else min(abs((d - j[0]).days), abs((d - j[1]).days))
    return 1 if dist(j1) <= dist(j2) else 2
