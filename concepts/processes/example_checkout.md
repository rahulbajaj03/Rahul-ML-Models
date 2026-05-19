# Checkout — Process Guide

## Overview
[One paragraph: what this process does end-to-end]

## Flow
```
User adds items to cart
    → Applies coupon (optional)
    → Selects payment method
    → Payment processed
    → Order confirmed
    → Fulfillment triggered
```

## Key decisions in the flow
| Decision point | Options | Impact on data |
|---------------|---------|---------------|
| Payment method | Card / UPI / Cash | Sets `payment_type` column |
| Coupon applied | Yes / No | Sets `discount_amount`, `coupon_id` |

## Where data is captured
| Step | Table | Key columns |
|------|-------|-------------|
| Order created | `orders` | `id`, `user_id`, `amount`, `status='pending'` |
| Payment | `payments` | `order_id`, `method`, `status` |
| Fulfillment | `shipments` | `order_id`, `shipped_at`, `delivered_at` |

## Common analysis questions
- What's the cart abandonment rate? → orders with status='pending' older than 1 hour
- What's the average order value? → SUM(amount)/COUNT(*) on completed orders
- Which payment method has highest failure rate? → payments WHERE status='failed' grouped by method

## Gotchas
- [Document anything non-obvious about this process]
