-- Matched controls for each focal work: same source, |year diff| <= k, |author-count diff| <= k,
-- article, no philosopher author, not itself a focal work. k in 1..{max_k} as nested scenarios.
-- Parameters: {slim_glob}, {focal}, {philosophers}, {max_k}, {bucket_cap}, {controls_per_focal}, {seed}
WITH focal AS (
    SELECT work_id, source_id, publication_year,
           coalesce(authors_count, n_authorships) AS n_authors
    FROM read_parquet('{focal}')
),
philosophers AS (SELECT author_id FROM read_parquet('{philosophers}')),
pool AS (
    SELECT work_id, source_id, publication_year,
           coalesce(authors_count, n_authorships) AS n_authors, author_ids
    FROM read_parquet('{slim_glob}', hive_partitioning = true)
    WHERE type = 'article'
      AND NOT coalesce(is_paratext, false)
      AND NOT coalesce(is_retracted, false)
      AND len(author_ids) > 0
      AND source_id IN (SELECT DISTINCT source_id FROM focal)
      AND publication_year BETWEEN (SELECT min(publication_year) FROM focal) - {max_k}
                               AND (SELECT max(publication_year) FROM focal) + {max_k}
),
-- works in the pool that have any philosopher author are not eligible controls
pool_with_phil AS (
    SELECT DISTINCT p.work_id
    FROM pool p, UNNEST(p.author_ids) AS u(author_id)
    JOIN philosophers ph ON u.author_id = ph.author_id
),
eligible AS (
    SELECT p.work_id, p.source_id, p.publication_year, p.n_authors
    FROM pool p
    ANTI JOIN pool_with_phil x ON p.work_id = x.work_id
    ANTI JOIN focal f ON p.work_id = f.work_id
),
-- cap each (source, year, n_authors) bucket to a random subset so the pair join stays bounded
bucketed AS (
    SELECT *, row_number() OVER (PARTITION BY source_id, publication_year, n_authors
                                 ORDER BY hash(work_id, {seed})) AS r
    FROM eligible
),
capped AS (SELECT work_id, source_id, publication_year, n_authors FROM bucketed WHERE r <= {bucket_cap}),
pairs AS (
    SELECT f.work_id AS focal_id, c.work_id AS control_id,
           abs(f.publication_year - c.publication_year) AS year_diff,
           abs(f.n_authors - c.n_authors)               AS author_diff,
           greatest(abs(f.publication_year - c.publication_year), abs(f.n_authors - c.n_authors)) AS min_k
    FROM focal f
    JOIN capped c
      ON c.source_id = f.source_id
     AND c.publication_year BETWEEN f.publication_year - {max_k} AND f.publication_year + {max_k}
     AND c.n_authors BETWEEN f.n_authors - {max_k} AND f.n_authors + {max_k}
),
-- one row per (focal, k, control) for every scenario k the pair qualifies for; sample per (focal, k)
ranked AS (
    SELECT p.*, k.k,
           row_number() OVER (PARTITION BY p.focal_id, k.k ORDER BY hash(p.focal_id, p.control_id, {seed})) AS rnk
    FROM pairs p
    JOIN range(1, {max_k} + 1) AS k(k) ON p.min_k <= k.k
)
SELECT focal_id, control_id, k, year_diff, author_diff
FROM ranked
WHERE rnk <= {controls_per_focal}
