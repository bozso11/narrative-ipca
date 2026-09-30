# Real data: input folder (placeholder)

This folder will hold the research pipeline's outputs for the dashboard's **Real data** page (DESIGN.md G.14, D82, D85; named "Real exposures" until 2026-09-29). It is empty until the pipeline delivers them. The page checks which files are present and previews the sensitivity table as soon as `sensitivities.parquet` exists.

**Topic sensitivity** (DESIGN.md G.0, D89): the expected return response of an asset to a one-standard-deviation attention shock in a topic, with the other topics' shocks held fixed. It is not a position size or dollar exposure. The file and column were named `exposures.parquet` and `exposure` until 2026-09-30; "exposure" in code identifiers still means topic sensitivity.

## Data contract (TBC)

| File | Content | Columns |
|---|---|---|
| `topics.csv` | One row per topic. | `topic_id`, `name`, `taxonomy_class`, `origin` (`fastopic` or `manual`), `model_version` |
| `attention.parquet` | Daily attention level per topic (FASTopic or manual topics on the news archive). | date index; one column per `topic_id` |
| `sensitivities.parquet` | Estimated topic sensitivity of each asset to each topic, per estimation date. One row per (`as_of`, `topic_id`, `asset_id`). | `as_of`, `topic_id`, `asset_id`, `sensitivity` (% return per one-sd attention shock), `uncertainty` (sd), `source` (`structural`, `regression`, `llm` or `blended`), `effective_window`, `coverage` (`full`, `partial` or `none`), `version` |

Assets and returns come from `data/reference/assets.csv` and `data/market/`; `asset_id` must match. The column names follow the interface fields of the research plan (sensitivity, uncertainty, source, effective window, coverage).

## Open points

1. Topic ids and versioning: plain ids (`T017`) or versioned ids (`T017@v3`) (TBC).
2. Whether `sensitivity` is the composed `B = M·L + S` only, or whether the factor route and the overlay are delivered separately (TBC).
