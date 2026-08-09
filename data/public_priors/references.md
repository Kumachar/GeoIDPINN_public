# Public prior references

The released matrices cover all 64 Louisiana county-equivalent units. Rows are
recipient counties `p`, columns are source counties `q`, and each `C0` row sums
to one.

- `county_metadata_with_centroids.csv`: Census county FIPS, names, 2020 Centers
  of Population population and coordinates.
- `distance_km.csv`: haversine distances from Census population centers.
- `adjacency_binary.csv`: Census 2025 county adjacency.
- `adjacency_shared_boundary_length_m.csv`: Census shared boundary length.
- `commute_jobs_work_by_home_LODES2019_*.csv`: public LODES OD employment flows
  aggregated from block to county.
- `C0_*.csv`: row-stochastic prior matrices constructed from those public
  inputs, including identity and negative-control priors.

Official sources:

- <https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt>
- <https://www2.census.gov/geo/docs/reference/county_adjacency/county_adjacency2025.txt>
- <https://lehd.ces.census.gov/data/lodes/LODES8/la/od/la_od_main_JT01_2019.csv.gz>
- <https://lehd.ces.census.gov/data/lodes/LODES8/la/od/la_od_main_JT00_2019.csv.gz>

The top-15 subset files from the internal analysis are intentionally excluded
because cohort selection used non-redistributed study observations.
