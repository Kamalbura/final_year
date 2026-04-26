## graphify

This project has a graphify knowledge graph at graphify-out/.

Mandatory policy:
- Before coding, run Graphify retrieval on the task.
- Use graphify query "<task/question>" --graph graphify-out/graph.json.
- If relation-specific, run graphify path "A" "B" --graph graphify-out/graph.json.
- Base edits only on returned nodes and relations.
- If confidence is low, ask one targeted follow-up query, not broad guessing.
- For multi-agent runs, every agent must use --graph graphify-out/graph.json.
- Keep the graph up to date: run graphify update . at session start, and graphify watch . during active development.
