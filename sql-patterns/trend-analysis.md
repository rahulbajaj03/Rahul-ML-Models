# Trend Analysis Patterns

## Day-over-Day comparison
```sql
SELECT 
    date,
    metric_value,
    LAG(metric_value) OVER (ORDER BY date) AS prev_day,
    ROUND((metric_value - LAG(metric_value) OVER (ORDER BY date)) * 100.0 
          / LAG(metric_value) OVER (ORDER BY date), 2) AS dod_change_pct
FROM your_daily_metrics
WHERE date >= DATE_ADD(CURRENT_DATE(), -14)
ORDER BY date
```

## 7-day rolling average
```sql
SELECT 
    date,
    metric_value,
    AVG(metric_value) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7d_avg
FROM your_daily_metrics
WHERE date >= DATE_ADD(CURRENT_DATE(), -30)
```

## Week-over-Week
```sql
SELECT 
    DATE_TRUNC('week', date) AS week_start,
    SUM(metric_value) AS weekly_total,
    LAG(SUM(metric_value)) OVER (ORDER BY DATE_TRUNC('week', date)) AS prev_week,
    ROUND((SUM(metric_value) - LAG(SUM(metric_value)) OVER (ORDER BY DATE_TRUNC('week', date))) * 100.0
          / LAG(SUM(metric_value)) OVER (ORDER BY DATE_TRUNC('week', date)), 2) AS wow_change_pct
FROM your_daily_metrics
GROUP BY DATE_TRUNC('week', date)
ORDER BY week_start
```
