# User — Entity Guide

## What is a user
[One paragraph: what this entity represents in your business]

## How a user enters the system
1. [Step 1 — e.g., signs up via app]
2. [Step 2 — e.g., verifies email]
3. [Step 3 — e.g., completes profile]
4. [Step 4 — e.g., makes first purchase]

## User identification
- **User ID:** Format `U` + number (e.g., `U12345`)
- **Other identifiers:** email, phone

## User statuses
| Status | Meaning |
|--------|---------|
| `active` | Can transact |
| `suspended` | Temporarily blocked |
| `churned` | Inactive > 30 days |

## Key relationships
| User connects to... | Via | Table |
|--------------------|-----|-------|
| Orders | `user_id` | `orders` |
| Payments | `user_id` | `payments` |

## Gotchas
- [Document anything non-obvious about this entity]
