# Data Landmines

> Every entry here exists because someone got burned. Read before writing any query.

## Timezone Traps

### [Example] createdAt is UTC, date partition is local
- `createdAt` columns are in **UTC**
- `date` partition columns are in **local timezone** (e.g., IST = UTC+5:30)
- A transaction at 11:30 PM local time on Jan 5 has `createdAt` = Jan 5 18:00 UTC but `date` = Jan 5
- **Rule:** Always filter on `date` partition for performance, use `createdAt` for precise timing

## Unit Traps

### [Example] Amounts are in cents/paise, not dollars/rupees
- `amount` columns store values in smallest unit (cents/paise)
- Divide by 100 for display: `amount / 100 AS amount_dollars`
- **Never** compare raw amounts to dollar values without converting

## Filter Traps

### [Example] Soft deletes — always filter deletedAt IS NULL
- Most tables use soft deletes
- Forgetting `deletedAt IS NULL` inflates counts by 10-30%
- **Rule:** Add `AND deletedAt IS NULL` to every query unless investigating deleted records

### [Example] Test/internal data
- Filter out test accounts: `WHERE user_id NOT LIKE 'TEST%'`
- Filter out internal users: `WHERE is_internal = false`

## Join Traps

### [Example] One-to-many explosion
- Joining orders to line_items multiplies rows
- Always aggregate AFTER joining, or use subqueries
- **Check:** Does your COUNT match expected volume? If 10x too high, you have a join explosion

## Add your own landmines below
<!-- Every time you or your team gets a wrong number, document it here -->
