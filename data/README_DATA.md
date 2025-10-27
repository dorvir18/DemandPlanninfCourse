
# Synthetic Demand Dataset (GitHub-ready)

## Files
- `products.csv` — master data for SKUs
  - `product_id` (str) — e.g. P001
  - `sku_name` (str)
  - `brand` (str) — one of 4 brands

- `clients.csv` — master data for clients
  - `client_id` (str) — e.g. C001
  - `client_name` (str)
  - `country` (str) — 10 countries
  - `region` (str) — 4 regions
  - `sales_manager` (str)

- `time_series.csv` — monthly per Client×SKU (Jan 2023–Sep 2025)
  - `client_id`, `product_id`
  - `date`, `year`, `month`
  - `shipments_to_client` — inbound shipments from manufacturer to client (отгрузки клиенту)
  - `shipments_from_client` — sell-out from client to shoppers (отгрузки от клиента)
  - `end_stock` — ending stock at client after sell-out
  - `latent_demand` — underlying demand before OOS constraint
  - `lost_sales` — unmet demand due to stock unavailability
  - `oos` — 1 if OOS evident (available=0 or lost_sales>0 or end_stock=0), else 0

## Consistency
`end_stock[t] = end_stock[t-1] + shipments_to_client[t] - shipments_from_client[t]`, bounded below by 0.
`sell-out` is capped by available stock. Seasonality exists for selected SKUs; several Client×SKU pairs have 2–3 month forced supply cuts creating OOS.

## License
Synthetic data for educational/demo purposes.
