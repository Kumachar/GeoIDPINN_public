# LODES commuting prior references

This folder includes LODES commuting priors produced from Census LEHD Origin-Destination Employment Statistics. The raw OD files are state-based and contain workplace block (`w_geocode`), residence block (`h_geocode`), and job counts (`S000`). The commuting priors aggregate the first five digits of each 15-digit block code to Louisiana parish FIPS codes. Rows are workplace/recipient parishes and columns are residence/source parishes.

Default files used by this script:

- `la_od_main_JT01_2019.csv.gz`: primary jobs.
- `la_od_main_JT00_2019.csv.gz`: all jobs.

Official source: U.S. Census Bureau LEHD/LODES downloads, https://lehd.ces.census.gov/data/lodes/.
