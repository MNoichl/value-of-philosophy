-- One pass over the OpenAlex works parquet: keep only what the pipeline needs.
-- Parameters are substituted by Python: {works_glob}, {deleted_ids}
SELECT
    w.id                                                    AS work_id,
    w.publication_year,
    w.publication_date,
    w.type,
    w.language,
    w.is_retracted,
    w.is_paratext,
    w.primary_location.source.id                            AS source_id,
    w.primary_location.source.issn_l                        AS issn_l,
    w.primary_location.source.type                          AS source_type,
    list_transform(w.authorships, lambda a: a.author.id)    AS author_ids,
    len(w.authorships)                                      AS n_authorships,
    w.authors_count,
    w.cited_by_count,
    w.counts_by_year,
    w.referenced_works_count,
    w.fwci,
    w.citation_normalized_percentile."value"                AS cnp,
    w.primary_topic.id                                      AS topic_id,
    w.primary_topic.subfield.id                             AS subfield_id,
    w.primary_topic.subfield.display_name                   AS subfield,
    w.primary_topic.field.display_name                      AS field,
    w.primary_topic.domain.display_name                     AS domain,
    w.updated_date
FROM read_parquet('{works_glob}', hive_partitioning = false) AS w
ANTI JOIN read_csv('{deleted_ids}', header = true) AS d
    ON w.id = d.work_id
