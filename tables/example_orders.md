# Table: `your_schema.orders`

## Overview
[One sentence: what does each row represent?]

**Partitioned by:** `date`

## Columns

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `id` | `int` | Primary key | Auto-increment |
| `user_id` | `varchar` | Foreign key to users | Format: `U` + number |
| `amount` | `int` | Order total | **In cents** — divide by 100 for dollars |
| `status` | `varchar` | Order status | Values: `pending`, `completed`, `cancelled`, `refunded` |
| `created_at` | `timestamp` | When order was placed | **UTC** |
| `date` | `date` | Date partition | **Local timezone** |
| `deleted_at` | `timestamp` | Soft delete | NULL = active. Always filter `deleted_at IS NULL` |

## Key relationships

| This column | Joins to | Join table |
|-------------|----------|-----------|
| `user_id` | `id` | `your_schema.users` |
| `id` | `order_id` | `your_schema.line_items` |

## Common queries

```sql
-- Daily order count (last 7 days)
SELECT date, COUNT(*) AS orders
FROM your_schema.orders
WHERE date >= DATE_ADD(CURRENT_DATE(), -7)
  AND deleted_at IS NULL
  AND status = 'completed'
GROUP BY date
ORDER BY date

-- Revenue by user (this month)
SELECT user_id, SUM(amount) / 100 AS revenue_dollars
FROM your_schema.orders
WHERE date >= DATE_TRUNC('month', CURRENT_DATE())
  AND deleted_at IS NULL
  AND status = 'completed'
GROUP BY user_id
ORDER BY revenue_dollars DESC
LIMIT 20
```

## Gotchas
- **Always filter `deleted_at IS NULL`** — soft deletes inflate counts
- **Amount is in cents** — divide by 100
- **`date` is local timezone, `created_at` is UTC** — don't mix them
- **Status `cancelled` vs `refunded`** — cancelled = never fulfilled, refunded = was fulfilled then reversed
