# AGENT_INDEX.md — Start every analytics session by reading this file

## What this repo is

This is your team's **agent flow** for analytics and business intelligence. It contains structured knowledge about your business, data model, metrics, SQL patterns, and known data gotchas — so that any AI assistant produces accurate queries and analysis.

## Always load these files

| File | What it contains |
|------|-----------------|
| `CONTEXT.md` | Business overview, domain vocabulary, entity relationships, data architecture |
| `LANDMINES.md` | Data gotchas that cause wrong numbers |

## Load by topic

### Concepts (business domain knowledge)
| If the question involves... | Load this file |
|----------------------------|---------------|
| Your core entity (e.g., users, orders) | `concepts/entities/your-entity.md` |
| Your core process (e.g., checkout, onboarding) | `concepts/processes/your-process.md` |

### Tables (schema documentation)
| If you need to query... | Load this file |
|------------------------|---------------|
| Your main table | `tables/your_table.md` |

### SQL Patterns
| If you need... | Load this file |
|---------------|---------------|
| Query standards | `sql-patterns/query-standards.md` |
| Trend analysis | `sql-patterns/trend-analysis.md` |

## Rules for AI assistants

1. **Clarify before querying** — if the question has ambiguity, ask before generating SQL
2. **Check LANDMINES.md** — before writing any query, scan for relevant gotchas
3. **Check the table documentation** — before querying any table, read its file for column meanings and filters

## Data environment

- **Platform:** [Your platform — Databricks / BigQuery / Snowflake / Postgres]
- **SQL dialect:** [Your dialect]
- **Schemas:** [Your schema structure]
