These are local prompt presets for OpenCode.

Why this exists:
- These files mirror your Claude agents in OpenCode's per-project agent format.
- OpenCode project agents live under `.opencode/agents/` and can be loaded as custom subagents when their frontmatter is valid.
- Subagents inherit the active session model when `model` is omitted, so these files intentionally do not set a model override.

How I will use them:
- `explorer-god.md` -> launch as custom subagent `explorer-god`
- `audit-god.md` -> launch as custom subagent `audit-god`
- `fintech-strategy-researcher.md` -> launch as custom subagent `fintech-strategy-researcher`

Read-only rule:
- These subagents now use OpenCode frontmatter permissions with `edit: deny` and `bash: deny`.
- They remain read-only explorers/researchers using file-reading/search tools and web fetch where available.

Planned review pattern for each project under `examples/`:
1. Architecture pass
2. Trading logic/risk pass
3. Reusable ideas and HydraQuant fit pass

That means I can run 3 parallel custom exploration agents per project using `explorer-god` once the runtime reloads the updated agent definitions.
