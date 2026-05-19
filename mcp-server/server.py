"""
Agent Flow MCP Server — Starter Template

Gives AI assistants:
- Knowledge base context (reads docs from this repo)
- Read-only SQL execution against your data warehouse
- Schema browsing

Requirements: pip install mcp databricks-sql-connector (or your DB connector)
"""

import os
import glob
from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("agent-flow")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@mcp.tool()
def get_context(topic: str) -> str:
    """
    Returns relevant knowledge base docs for a topic.
    Call this FIRST before writing any query.
    """
    results = []
    
    # Search all markdown files for the topic
    for md_file in glob.glob(os.path.join(REPO_ROOT, "**/*.md"), recursive=True):
        with open(md_file, "r") as f:
            content = f.read()
        if topic.lower() in content.lower():
            results.append(f"--- {os.path.relpath(md_file, REPO_ROOT)} ---\n{content[:2000]}")
    
    if not results:
        return f"No docs found for topic: {topic}. Try a different keyword."
    
    return "\n\n".join(results[:5])  # Return top 5 matches


@mcp.tool()
def run_query(sql: str) -> str:
    """
    Execute read-only SQL query. SELECT/SHOW/DESCRIBE only.
    Returns up to 500 rows.
    """
    # Safety check
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
    sql_upper = sql.upper().strip()
    for word in forbidden:
        if sql_upper.startswith(word):
            return f"ERROR: {word} statements are not allowed. Read-only access only."
    
    # --- REPLACE THIS WITH YOUR DATABASE CONNECTION ---
    # Example for Databricks:
    # from databricks import sql as databricks_sql
    # connection = databricks_sql.connect(
    #     server_hostname=os.environ["DATABRICKS_HOST"],
    #     http_path=os.environ["DATABRICKS_HTTP_PATH"],
    #     access_token=os.environ["DATABRICKS_TOKEN"],
    # )
    # cursor = connection.cursor()
    # cursor.execute(sql)
    # columns = [desc[0] for desc in cursor.description]
    # rows = cursor.fetchmany(500)
    # return format_results(columns, rows)
    
    return "DATABASE NOT CONFIGURED — replace the connection code in server.py"


@mcp.tool()
def search_knowledge(query: str) -> str:
    """Keyword search across all knowledge base files."""
    results = []
    for md_file in glob.glob(os.path.join(REPO_ROOT, "**/*.md"), recursive=True):
        with open(md_file, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if query.lower() in line.lower():
                context = "".join(lines[max(0, i-2):i+3])
                results.append(f"{os.path.relpath(md_file, REPO_ROOT)}:{i+1}\n{context}")
    
    if not results:
        return f"No matches for: {query}"
    return "\n---\n".join(results[:10])


if __name__ == "__main__":
    mcp.run()
