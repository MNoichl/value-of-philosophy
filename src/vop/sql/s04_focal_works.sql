-- Focal works: articles with >= 1 philosopher author, published in a NON-philosophy source.
-- Parameters: {slim_glob}, {phil_sources}, {philosophers}, {year_min}, {year_max}, {excluded_subfields}
WITH phil_src AS (SELECT source_id FROM read_parquet('{phil_sources}')),
philosophers AS (SELECT author_id FROM read_parquet('{philosophers}')),
articles AS (
    SELECT *
    FROM read_parquet('{slim_glob}', hive_partitioning = true)
    WHERE type = 'article'
      AND NOT coalesce(is_paratext, false)
      AND NOT coalesce(is_retracted, false)
      AND publication_year BETWEEN {year_min} AND {year_max}
      AND source_id IS NOT NULL
      AND len(author_ids) > 0
),
-- per work: the philosopher authors, in author order
flagged AS (
    SELECT a.work_id, list(u.author_id ORDER BY u.i) AS phil_author_ids
    FROM articles a, UNNEST(a.author_ids) WITH ORDINALITY AS u(author_id, i)
    JOIN philosophers p ON u.author_id = p.author_id
    GROUP BY a.work_id
)
SELECT
    a.*,
    f.phil_author_ids,
    len(f.phil_author_ids)                               AS n_philosophers,
    list_contains(f.phil_author_ids, a.author_ids[1])    AS first_author_is_phil,
    list_contains(f.phil_author_ids, a.author_ids[-1])   AS last_author_is_phil
FROM articles a
JOIN flagged f USING (work_id)
ANTI JOIN phil_src p ON a.source_id = p.source_id
WHERE coalesce(a.subfield, '') NOT IN ({excluded_subfields})
