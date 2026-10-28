# OpenAlex parquet snapshot notes

Snapshot in use: release date **2026-10-14** (`data/parquet/works/manifest.json`: 1,898 files,
464,765,485 records, 744 GB). Partitioned by `updated_date=YYYY-MM-DD/part_NNNN.parquet`.

- `schema_works.csv`, `schema_sources.csv`: output of DuckDB `DESCRIBE` on the real files. All SQL in
  `src/vop/sql/` is written against these column names.
- Useful `works` columns for us: `authors_count`, `authorships[].author.id`, `primary_location.source.{id,issn_l,issn}`,
  `primary_topic.{subfield,field,domain}`, `cited_by_count`, `counts_by_year[]` (year, cited_by_count),
  `fwci`, `citation_normalized_percentile`, `referenced_works_count`, `is_retracted`, `is_paratext`, `type`, `language`.
- The AWS CLI must be given `--region us-east-1` (or `AWS_DEFAULT_REGION=us-east-1`); the SRC image otherwise resolves a bogus region.
- Download: `aws s3 sync s3://openalex/data/parquet/<entity> data/raw/openalex/data/parquet/<entity> --no-sign-request --region us-east-1`
