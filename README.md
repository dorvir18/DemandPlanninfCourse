# Demand Forecast Lab (GitHub Pages + Actions)

Стек:
- **/data** — исходные CSV (клиенты, продукты, помесячные ряды Client×SKU)
- **/pipelines/build_artifacts.py** — Python-пайплайн (SES/ARIMA, метрики)
- **/docs** — сайт для GitHub Pages (интерактивная страница, артефакты в /docs/data)
- **.github/workflows/build.yml** — GitHub Actions: пересчёт артефактов при изменениях/по расписанию

## Быстрый старт
1) Создайте репозиторий на GitHub и включите **Pages**: Settings → Pages → `Branch: main`, `Folder: /docs`  
2) Скопируйте файлы из этого архива.  
3) В разделе Actions запустите **Build artifacts** (`Run workflow`) или сделайте commit.  
4) Откройте `<ваш-юзер>.github.io/<repo>/` — на главной странице появится список пар Client×SKU из `catalog.json` и график прогноза.

## Данные
- `data/products.csv`, `data/clients.csv` — справочники (бренд/страна/регион/менеджер)
- `data/time_series.csv` — ежемесячные факты: shipments_to_client, shipments_from_client (sell-out), end_stock, latent_demand, lost_sales, oos

## Артефакты
- `docs/data/forecast_{client_id}_{product_id}.json`
- `docs/data/metrics_{client_id}_{product_id}.json`
- `docs/data/catalog.json` — пары для интерфейса

## Локальный тест
```
pip install -r requirements.txt
python pipelines/build_artifacts.py
# Откройте docs/index.html в браузере (или поднимите локальный http-сервер)
```

## Примечания
- Страница использует uPlot (без eval) — дружелюбно к CSP.
- Модели и метрики минимальные — расширяйте под уроки (ETS, SARIMA, ARIMAX, Prophet и т.д.).
