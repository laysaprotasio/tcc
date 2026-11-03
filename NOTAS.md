# NOTAS — replicação do Confident Learning (Miyaji, Almeida e Corrêa)

Estado: **bloqueado na etapa 2** (API de ocorrências do GBIF fora do ar em 02–03/11/2026; ver "Pendências").

## Fontes
- O PDF do SBBD 2023 **não está na pasta**; usei a versão estendida (JIDM 2025, doi 10.5753/jidm.2025.4230) colada na conversa. Os alvos numéricos do SBBD (Tabela 2, t0/t1, Q) vieram do enunciado.
- Divergência dentro do próprio artigo: "proporção 1:3" implica 25% de positivos, mas o texto diz ~23%. O código usa 1:3 (`--smote-ratio 0.3333`), configurável.
- JIDM, Tabela 2: Q = [[.624,.167],[.073,.136]]; enunciado (SBBD): [[.624,.169],[.073,.134]]. Diferença irrelevante.

## Ambiente
- `.venv` com Python 3.14.3; pygbif 0.7.0, pandas 3.1.0, scikit-learn 1.9.1, imbalanced-learn 0.14.2, cleanlab 2.9.0.
- Sementes: 42, 43, … (`--seeds N`).
- Comando único: `.venv\Scripts\python.exe run_all.py` (`--pular-gbif` reaproveita `data/gbif_raw.csv`).

## Perfil do CSV (resultado real, `results/perfil_csv.json`)
- 422.649 linhas; colunas têm unidade no cabeçalho (ex.: "CO (ppm)"); removo o sufixo.
- 8 preditores: CO, O3, NOX, CPC3010, Acetonitrile, Isoprene, CH4, Temperature. **Não existem** Tmax/Tmin, CO2, H2O (o artigo cita); uso uma única temperatura.
- Faltantes: CO 6,55%; O3 7,02%; NOX 9,40%; CPC3010 10,48%; Acetonitrile e Isoprene 9,96%; CH4 0,29%; Temperature 8,52%. Linhas completas: 366.020 (IOP1: 183.194; IOP2: 182.826).
- **A grade não é um raster regular**: 748 latitudes × 1563 longitudes distintas, mas 422.649 pontos únicos (um por lat/lon/IOP); passo mediano 0,0015° em lat e 0,0006° em lon. Parece ser trajetória interpolada dos voos, não cobertura de área.
- Hipótese (não verificada): as 185.305 linhas do artigo ≈ linhas completas de **uma** IOP (183.194 ou 182.826; diferença ~1%). Por isso `04_rotulos.py` gera a tabela para `1+2`, `1` e `2`, e `--iops` é configurável.
- Correlações |r|≥0,80 já visíveis: CO–O3 0,91, CO–Temp 0,94, CO–Acetonitrile 0,86, NOX–CPC3010 0,87, O3–Temp 0,83. Tende a remover mais de 3 variáveis (o artigo removeu 3) — verificar depois.

## Suposições de projeto (a validar)
1. Janelas: IOP1 = 01/02–31/03/2014; IOP2 = 15/08–15/10/2014 (dia exato, via `eventDate`). Registros sem data válida ou fora da janela são descartados (contagem em `results/registros_etapas.json`). Ano 2015 é buscado, mas o CSV só tem 2014, logo 2015 cai na filtragem.
2. Junção: ponto mais próximo (distância planar local, m) entre as células completas da mesma IOP; registros a mais de `--max-match-m` (padrão 5 km, escolha minha) ficam sem par.
3. Positivos: célula mais próxima + células da mesma IOP dentro do raio. Raio escolhido automaticamente como o que deixa a proporção mais perto de 3,3%. **Isso é calibrar um parâmetro livre contra o resultado publicado** — não valida o artigo, só o aproxima; relatar assim no TCC.
4. Negativos: todas as demais células (inclusive de outra IOP). Presence-only ⇒ ausência incerta.
5. Faltantes: remoção das linhas incompletas (sem imputação) ⇒ 366.020 linhas com ambas IOPs.
6. Correlação: remoção iterativa da variável com mais pares |r|≥0,80 (desempate: maior |r| médio); calculada no conjunto todo (não usa o rótulo).
7. Modelo: StandardScaler + LogisticRegression(C=1), limiar 0,5. A padronização não é descrita no artigo.
8. CL: `prune_by_noise_rate`; t_j = média das probabilidades fora da amostra da classe j entre os rotulados j (Eq. 3); Q via `cleanlab.count.estimate_joint`.
9. Ordem do protocolo do artigo (A): SMOTE no conjunto todo → CL → CV 5-fold aleatória no conjunto podado. SMOTE antes da CV vaza sintéticos entre folds; foi mantido de propósito para replicar.
10. Avaliações adicionais: B1 = StratifiedGroupKFold por blocos de 0,05° (SMOTE e CL só no treino; teste com rótulos originais, sem poda, sem SMOTE; folds sem positivo no teste são pulados e contados); B2 = holdout aleatório 25% com as mesmas regras.

## Pendências / problemas
- **GBIF**: `api.gbif.org/v1/occurrence/search` devolve 503 para qualquer consulta (inclusive `?limit=1`), enquanto `species/match` e `occurrence/count` respondem 200. O `01_gbif.py` tenta de novo por ~40 min e esgotou. Alternativa: pedir **download com DOI** no site do GBIF (filtros: *Coragyps atratus*, 2014, polígono do CSV, com coordenadas) e salvar como `data/gbif_raw.csv`.
- **SiBBr/ICMBio**: `biocache.sibbr.gov.br` devolveu 502; não confirmei endpoint válido. `03_registros.py` tenta e segue só com GBIF se falhar.
- Comparação com os 54 registros e o critério de parada (±50%) só podem ser avaliados depois que houver dados reais.
- Tempo: um teste de fumaça com presenças **fictícias** (só no scratchpad, nunca em `results/`) levou >10 min por semente; 10 sementes podem passar de 1–2 h. Pode ser preciso subsample/solver mais rápido.
