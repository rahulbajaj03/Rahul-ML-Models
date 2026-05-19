# Query Standards

## Before writing any query

1. **Clarify ambiguity** — don't assume. Ask about:
   - Date range (last 7 days? last month? all time?)
   - Filters (include test users? include cancelled orders?)
   - Granularity (daily? weekly? per-user?)

2. **Check LANDMINES.md** — scan for relevant gotchas before writing

3. **Check table docs** — read the table's `.md` file for column meanings

## Query structure

```sql
-- Always comment what this query answers
SELECT 
    dimension_columns,
    AGG(metric_columns) AS clear_alias_name
FROM schema.table
WHERE date_partition_filter          -- always filter partition first
  AND deleted_at IS NULL             -- soft delete filter
  AND business_logic_filters         -- status, type, etc.
GROUP BY dimension_columns
ORDER BY meaningful_order
LIMIT reasonable_number              -- don't pull millions of rows
```

## Naming conventions
- Use `snake_case` for aliases
- Name aggregations clearly: `total_orders`, `avg_amount_dollars`, `pct_cancelled`
- Don't use `x`, `t1`, `t2` — use meaningful table aliases

## Common patterns

### Date filtering
```sql
-- Last 7 days
WHERE date >= DATE_ADD(CURRENT_DATE(), -7)

-- This month
WHERE date >= DATE_TRUNC('month', CURRENT_DATE())

-- Specific range
WHERE date BETWEEN '2024-01-01' AND '2024-01-31'
```

### Safe aggregation (avoid join explosions)
```sql
-- BAD: joining then counting
SELECT u.id, COUNT(*) FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id

-- GOOD: aggregate in subquery first
SELECT u.id, o.order_count
FROM users u
JOIN (SELECT user_id, COUNT(*) AS order_count FROM orders GROUP BY user_id) o
  ON u.id = o.user_id
```
