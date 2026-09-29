# Real exposures: input folder (placeholder)

This folder will hold the research pipeline's outputs for the dashboard's **Real exposures** page (DESIGN.md G.14, D82). It is empty until the pipeline delivers them. The page checks which files are present and previews the exposure table as soon as `exposures.parquet` exists.

## Data contract (TBC)

| File | Content | Columns |
|---|---|---|
| `topics.csv` | One row per topic. | `topic_id`, `name`, `taxonomy_class`, `origin` (`fastopic` or `manual`), `model_version` |
| `attention.parquet` | Daily attention level per topic (FASTopic or manual topics on the news archive). | date index; one column per `topic_id` |
| `exposures.parquet` | Estimated exposure of each asset to each topic, per estimation date. One row per (`as_of`, `topic_id`, `asset_id`). | `as_of`, `topic_id`, `asset_id`, `exposure` (% return per one-sd attention shock), `uncertainty` (sd), `source` (`structural`, `regression`, `llm` or `blended`), `effective_window`, `coverage` (`full`, `partial` or `none`), `version` |

Assets and returns come from `data/reference/assets.csv` and `data/market/`; `asset_id` must match. The column names follow the interface fields of the research plan (exposure, uncertainty, source, effective window, coverage).

## Open points

1. Topic ids and versioning: plain ids (`T017`) or versioned ids (`T017@v3`) (TBC).
2. Whether `exposure` is the composed `B = M·L + S` only, or whether the factor route and the overlay are delivered separately (TBC).
