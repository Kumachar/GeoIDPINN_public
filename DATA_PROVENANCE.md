# Data provenance and release policy

## Public input data

### U.S. Census 2020 Centers of Population

- Official source: <https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt>
- Released fields: county FIPS, county name, state, population, latitude, and
  longitude.
- Transformation: Louisiana county-equivalent rows were selected and pairwise
  haversine distances were computed.

### U.S. Census 2025 County Adjacency File

- Official source: <https://www2.census.gov/geo/docs/reference/county_adjacency/county_adjacency2025.txt>
- Transformation: Louisiana county-equivalent pairs were selected; adjacency
  and shared-boundary matrices were assembled in a fixed 64-county order.

### U.S. Census LEHD LODES8, 2019

- Primary jobs (JT01): <https://lehd.ces.census.gov/data/lodes/LODES8/la/od/la_od_main_JT01_2019.csv.gz>
- All jobs (JT00): <https://lehd.ces.census.gov/data/lodes/LODES8/la/od/la_od_main_JT00_2019.csv.gz>
- Transformation: workplace and residence block GEOIDs were aggregated to
  five-digit county FIPS and converted to row-stochastic coupling priors.

### CDC weekly state cases

- Official archived dataset: <https://data.cdc.gov/Case-Surveillance/Weekly-United-States-COVID-19-Cases-and-Deaths-by-/pwn4-m3yp>
- Released subset: Arkansas, Mississippi, and Texas weekly new cases for the
  2020 analysis interval.
- Transformation: the three state series were aligned by week for the observed
  external infection-pressure covariate.

## Derived research outputs

`results/tables/` contains only model-level or experiment-level aggregate
metrics. `results/matrices/` contains the learned coupling matrices averaged
over 12 fits (four forecast origins by three seeds), plus an aggregate matrix
distance summary. These are research results, not source observations.

The paper trajectory panel is a deliberately released publication artifact.
Its underlying county-week observed/predicted table is not included.

## Data intentionally excluded

- the original Louisiana tract-level Stata file;
- county-week case and testing tables;
- top-15 cohort metadata containing observed totals;
- per-origin or per-seed predictions and latent states;
- observed outside-signal exports generated during training;
- checkpoints, optimizer states, and per-run learned matrices;
- notebook outputs containing local paths or tabular observations.

The original Louisiana study data must be obtained from its data steward. Its
license, permanent citation, access terms, and redistribution permission must
be confirmed independently before any raw or row-level data are published.
