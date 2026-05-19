# Business Context

## What we do
[One paragraph about your business]

## Key entities
- **Users/Customers:** [what they are, ID format]
- **Orders/Transactions:** [what they are, lifecycle]
- **Products/Services:** [what you sell/provide]

## Entity relationships
```
User → places → Order → contains → LineItems → references → Product
```

## Key metrics
| Metric | Definition | Formula |
|--------|-----------|---------|
| GMV | Gross Merchandise Value | SUM(order_amount) |
| [Your metric] | [Definition] | [Formula] |

## Data architecture
- Raw data lands in: [schema/database]
- Processed data in: [schema/database]
- Timezone convention: [UTC / local / mixed — specify per table]
- Currency unit: [dollars / cents / paise — specify]

## Teams that use this data
- [Team 1] — cares about [metrics]
- [Team 2] — cares about [metrics]
