# Data directory

The **Prediction** tab's "Select Data File" dropdown (`/api/data-files`) scans this folder for
`.csv` and `.feather` files, so it needs at least one file present to have anything to list.

Drop your own K-line data here to use it — required columns: `open`, `high`, `low`, `close`.
Optional: `volume`, `amount`, and a timestamp column named `timestamps`, `timestamp`, or `date`.

`sample_5min_kline.csv` is real 5-minute A-share K-line data (2,500 rows) included so the
dropdown always has at least one working example out of the box.
