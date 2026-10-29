-- Per-author publication counts, restricted to authors with >= 1 article in a philosophy source.
-- Parameters: {slim_glob}, {phil_sources}, {year_min}
WITH articles AS (
    SELECT work_id, publication_year, source_id, author_ids
    FROM read_parquet('{slim_glob}', hive_partitioning = true)
    WHERE type = 'article'
      AND NOT coalesce(is_paratext, false)
      AND NOT coalesce(is_retracted, false)
      AND publication_year >= {year_min}
      AND len(author_ids) > 0
),
phil AS (
    SELECT source_id FROM read_parquet('{phil_sources}')
),
-- candidate authors: anyone with at least one article in a philosophy source
candidates AS (
    SELECT DISTINCT u.author_id
    FROM articles a
    SEMI JOIN phil p ON a.source_id = p.source_id,
    UNNEST(a.author_ids) AS u(author_id)
),
-- all articles of those candidates, one row per (work, author), deduplicated within a work
author_works AS (
    SELECT DISTINCT u.author_id, a.work_id, a.source_id, a.publication_year
    FROM articles a, UNNEST(a.author_ids) AS u(author_id)
    SEMI JOIN candidates c ON u.author_id = c.author_id
)
SELECT
    aw.author_id,
    count(*)                                                   AS works_total,
    sum(CASE WHEN p.source_id IS NOT NULL THEN 1 ELSE 0 END)   AS works_phil,
    works_phil / works_total                                   AS share_phil,
    min(aw.publication_year)                                   AS first_year,
    max(aw.publication_year)                                   AS last_year
FROM author_works aw
LEFT JOIN phil p ON aw.source_id = p.source_id
GROUP BY aw.author_id
