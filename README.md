# Agent Flow — AI Analytics Agent Template

A structured knowledge repo that makes AI assistants (Claude, Cursor, Kiro, ChatGPT) produce accurate SQL queries and analysis for YOUR data.

## The Problem

AI assistants hallucinate table names, miss timezone traps, forget unit conversions, and silently include data that should be filtered. Every wrong answer erodes trust.

## The Solution

Document your data knowledge in a structured format that AI can read before answering. Think of it as your team's shared memory.

## Structure

```
agent-flow-template/
├── AGENT_INDEX.md              ← AI reads this first (routing table)
├── CONTEXT.md                  ← Business overview, entities, architecture
├── LANDMINES.md                ← Data gotchas (the stuff that burns you)
├── tables/                     ← One file per table (schema + gotchas + examples)
│   └── example_orders.md
├── concepts/
│   ├── entities/               ← What things ARE (users, products, etc.)
│   │   └── example_user.md
│   └── processes/              ← How things WORK (checkout, onboarding, etc.)
│       └── example_checkout.md
├── sql-patterns/               ← Reusable query templates
│   ├── query-standards.md
│   └── trend-analysis.md
└── mcp-server/                 ← Optional: gives AI direct SQL access
    ├── server.py
    └── requirements.txt
```

## Quick Start

1. Clone this template
2. Replace example files with your own tables/entities/processes
3. Start with LANDMINES.md — document every data gotcha you know
4. Add table docs one at a time (start with your most-queried tables)
5. Point your AI tool at this repo

### Using with Kiro/Cursor (file-based)
Just open this repo in your IDE. The AI reads the files directly.

### Using with Claude/ChatGPT (MCP server)
```bash
cd mcp-server
pip install -r requirements.txt
# Configure your database connection in server.py
python server.py
```

## How to Grow It

Every time you or your team:
- Gets a wrong number → add to LANDMINES.md
- Discovers a non-obvious column meaning → add to the table doc
- Explains a business process to someone → write it as a process doc
- Writes a useful query pattern → add to sql-patterns/

It compounds. The agent never makes the same mistake twice.

## Tips

- Write for an AI reader: be explicit, not implicit
- Include "Gotchas" sections in every file
- Add common queries with comments explaining WHY each filter exists
- Document what columns DON'T mean (common misinterpretations)
- Keep it in git — version control your team's knowledge
