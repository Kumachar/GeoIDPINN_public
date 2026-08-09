# Released data

`public_priors/` contains only public U.S. Census geography and LODES-derived
inputs for all 64 Louisiana county-equivalents. The public-facing metadata file
is `county_metadata_with_centroids.csv`. The smaller
`parish_metadata_with_centroids.csv` is a compatibility view used by the
original experiment scripts; it contains the same public fields and no case or
testing summaries.

`external/` contains a public CDC state-level weekly series. See
`../DATA_PROVENANCE.md` for sources and transformations.

No Louisiana tract-level or county-week study observations are included.
