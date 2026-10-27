# Design Decisions and Potential Improvements

This document summarizes the system's design choices and potential
improvements.

## Design Decisions

### Indexing

- **Use FashionCLIP for image and text embeddings.** A pretrained
  fashion-oriented model provides shared visual and semantic features.
- **Prefer higher-resolution image columns.** Select the first column with
  valid URLs in large-to-small order; skip failed URLs within that column.
- **Pool product image views.** Average successful image embeddings into one
  normalized vector per product.
- **Build structured product text.** Embed available product name, brand,
  color, category, and keywords with field labels.
- **Index modalities independently.** Separate image and text indexes allow
  either modality to retrieve candidates. Products with numeric data can
  remain in metadata even without vectors.
- **Use HNSW for approximate retrieval.** It reduces search work compared with
  exhaustive nearest-neighbor search, with a trade-off between speed and
  recall.
- **Build indexes and metadata together.** Shared IDs require all artifacts
  to come from the same indexing run.

### Retrieval and Ranking

- **Generate query embeddings at request time.** Fresh catalog content is
  used, at the cost of repeating image downloads and model inference.
- **Retrieve by modality, then combine candidates.** Image and text search
  results are unioned for reranking; a missing modality match scores zero.
- **Normalize vector similarities.** Convert cosine similarity to a common
  `[0, 1]` scale for ranking.
- **Combine visual, semantic, and numeric signals with fixed weights.**
- **Keep weights fixed for missing signals.** Unavailable features score zero;
  weights are not redistributed across available features.
- **Use numeric fallback when vector retrieval is empty.** It scans indexed
  metadata for numeric matches, but does not expand a non-empty vector
  candidate set.
- **Allow queries for catalog-only products.** A query product need not have
  an index entry if it exists in the source catalog.

## Potential Improvements

### Architecture Improvements

- **Cache repeated work.** Cache query embeddings or recommendation results
  with clear invalidation when catalog, index, or image data changes.
- **Fall back across image sizes after download failures.** Try medium and
  small URLs if all selected large-image URLs fail, consistently in indexing
  and query-time embedding.
- **Improve observability.** Track request and inference latency, image
  download failures, candidate counts, fallback usage, and empty results.
- **Control concurrency.** Limit concurrent downloads and model inference to
  protect compute resources and keep latency predictable.
- **Scale numeric fallback.** Use bounded numeric candidate generation and
  merge those candidates with vector results where appropriate.
- **Validate artifacts and reliability.** Check index/metadata compatibility
  at startup and test missing data, failed downloads, and concurrent load.

### Accuracy Improvements

- **Tune HNSW parameters.** Measure recall and latency across search settings
  and candidate-pool sizes; choose a configuration that meets quality and
  performance targets.
- **Evaluate dynamic weights.** Compare fixed weights with weights
  renormalized over available attributes, especially for products with
  incomplete data.
- **Measure attribute importance.** Use relevance judgments or interaction
  data to assess the contribution of image, text fields, price, weight, and
  rating; adjust features based on results.
- **Improve candidate coverage.** Evaluate adding numeric candidates to
  vector results and measure the impact on recall.
- **Use catalog-aware constraints where helpful.** Evaluate category,
  availability, and variant handling to reduce irrelevant matches.
- **Calibrate score interpretation.** Treat the current score as a ranking
  signal; calibrate it against relevance data before interpreting it as a
  probability.
