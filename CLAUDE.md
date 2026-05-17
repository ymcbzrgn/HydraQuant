# Memory & Knowledge Systems — Operating Manual

Bu projede aktif olan memory/bilgi katmanları ve **nasıl kullanılacaklarına** dair çalışma talimatı. Proje-spesifik gerçekler `MEMORY.md` ve altındaki dosyalarda. Burası sadece **araçların doğru kullanım rehberi**.

---

## Aktif Katmanlar

### 1. Auto-memory (built-in, otomatik yüklenir)
- Konum: `~/.claude/projects/<project-id>/memory/`
- `MEMORY.md` her SessionStart'ta context'e enjekte edilir → ekranda zaten var
- Tipler: `feedback_*.md`, `project_*.md`, `reference_*.md`, `user_*.md`
- Dosyalar `[[name]]` ile cross-link

### 2. claude-mem plugin (cross-session kalıcı hafıza)
- SQLite (`~/.claude-mem/claude-mem.db`) + Chroma vector DB + Worker (port 37777)
- SessionStart'ta `$CMEM` blok'u inject olur: recent sessions (S###), recent observations (####), token sayaçları
- Viewer UI: http://localhost:37777
- **MCP araçları** (`mcp__plugin_claude-mem_mcp-search__*`):
  - `search`, `timeline`, `get_observations` — 3 katmanlı arama
  - `smart_search`, `smart_outline`, `smart_unfold` — AST-tabanlı kod keşfi
  - `build_corpus`, `prime_corpus`, `query_corpus`, `list_corpora`, `rebuild_corpus`, `reprime_corpus` — temalı bilgi tabanı
- **Skill'ler**: `mem-search`, `smart-explore`, `make-plan`, `do` (claude-mem:do), `timeline-report`, `knowledge-agent`

### 3. code-review-graph (project MCP, auto-update)
- Persistent incremental code knowledge graph (Python/TS/JS/Vue parsing, FTS5)
- Hook'lar (`.claude/settings.json`):
  - **PostToolUse** (Edit|Write|Bash) → `code-review-graph update --skip-flows` çalışır, graf güncellenir
  - **SessionStart** → `code-review-graph status` yazdırır (Nodes/Edges/Files/Languages özet)
  - **PreToolUse** (Glob|Grep) → `graphify-out/GRAPH_REPORT.md` varsa hatırlatma context'i inject eder
- Binary: `<project>/.venv/bin/code-review-graph`
- Manuel komutlar (gerekirse): `status | visualize | wiki | detect-changes | eval`

### 4. graphify (one-shot, talep üzerine)
- Trigger: `/graphify <path>` (yoksa `.`)
- Output: `graphify-out/graph.json` + `GRAPH_REPORT.md` + interactive HTML + Obsidian vault
- Her edge `EXTRACTED | INFERRED | AMBIGUOUS` tag'iyle audit edilir
- `--update` ile incremental, `--mcp` ile MCP server modu
- PreToolUse hook bu output varsa Glob/Grep öncesi okumayı zorlar

### 5. Custom subagents
- `.claude/agents/*.md` — proje-spesifik uzman ajanlar (Opus, anti-hallucination protokolüyle)
- `.claude/agent-memory/<agent-name>/` — ajan başına persistent state (her ajan kendi öğrendiğini orada tutar)
- Agent tool ile `subagent_type: <name>` parametresiyle çağrılır

---

## Karar Ağacı — Hangisini Ne Zaman?

### A. Task başlangıcı / "bunu daha önce yaptık mı?"
1. `MEMORY.md` zaten yüklü — **ilk durak**
2. `$CMEM` listesinde ilgili S### veya #### ID'leri varsa → `get_observations(ids=[…])` ile detayını aç (sadece title okuma, içeriği al)
3. Hiç eşleşme yoksa → `mem-search` skill veya MCP `search(query, limit, type, dateStart)` → relevant ID'leri seç → `timeline(anchor=ID)` ile çevresini al → `get_observations([IDs])` ile genişlet
4. Tematik derinlik gerekiyorsa → `knowledge-agent`: `build_corpus` → `prime_corpus` → `query_corpus`

### B. Kod keşfi (function/symbol/file bulma)
**Default: smart-explore. Read/Grep/Glob refleksini bastır.**
- `smart_search(query, path, max_results)` — symbol + file discovery tek çağrıda
- `smart_outline(file_path)` — file iskeleti (imports + signatures)
- `smart_unfold(file_path, symbol_name)` — tek symbol full source

Read/Grep/Glob'a sadece şu durumlarda dön:
- Config/JSON/YAML/SQL/log dosyaları (AST parse edilmiyor)
- Tree-sitter dil desteği dışında bir dosya
- Tam dosya içeriği gerekiyor (Read), regex match gerekiyor (Grep)

### C. Mimari / cross-module ilişki / "X nerede kullanılıyor?"
1. `graphify-out/GRAPH_REPORT.md` varsa **önce o** — community map + god nodes
2. code-review-graph status/wiki/visualize ile node-edge sorgu (binary mevcutsa)
3. Derin verified analiz için → `explorer-god` agent (Opus, fact-only, file:line zorunlu)

### D. Code audit / change review
- `audit-god` agent (Opus, abstention protocol, chain-of-verification)
- Tüm bulgular file:line evidence ile

### E. Multi-step plan/implementation
- `/make-plan` (claude-mem:make-plan) → Phase 0 documentation discovery + phased plan
- `/do` (claude-mem:do) → plan execution with subagents
- Tek-shot küçük iş için skill'leri bypass et, doğrudan yap

### F. Project history narrative / dönemsel rapor
- `/timeline-report` skill

### G. Tekrarlayan / scheduled iş
- `/loop` skill (interval'lı)
- `/schedule` skill (cron-based remote agent)

---

## Auto-Fire Trigger'ları (Hook Davranışları)

| Olay | Sistem otomatik yapar | Senin görevin |
|---|---|---|
| **SessionStart** | `$CMEM` inject + `code-review-graph status` | `MEMORY.md` ve `$CMEM`'i oku, ilgili ID'leri akılda tut |
| **PostToolUse** (Edit/Write/Bash) | `code-review-graph update --skip-flows` | Müdahale yok — hook hallediyor |
| **PreToolUse** (Glob/Grep) | `graphify-out/GRAPH_REPORT.md` hint inject | Varsa rapor'u oku, sonra Glob/Grep |
| **Skill çağrısı** | Skill markdown'u context'e gelir | Skill talimatlarını birebir takip et |
| **`<private>...</private>` etiketi** | claude-mem o içeriği DB'ye yazmaz | Sensitive bilgi (key/token/şifre) için zorunlu |

---

## Memory'ye Ne Zaman Yazılır?

### Yaz
| Durum | Tip | Örnek |
|---|---|---|
| User correction ("şöyle yapma") | `feedback_*.md` | "Mock DB kullanma, prod migration kırılmıştı" |
| User confirmation non-obvious'a ("evet böyle iyi") | `feedback_*.md` | "Tek bundled PR doğru tercihti — split etme" |
| Project state / deadline / constraint | `project_*.md` | "Merge freeze 2026-03-05 başlıyor" |
| External system referansı | `reference_*.md` | "Bug'lar Linear INGEST'te" |
| User role / persistent preference | `user_*.md` | "Data scientist, observability odaklı" |

### Yazma
- Kod path/pattern (kodu okuyarak bulunur — git ve dosya zaten source of truth)
- Git history / commit messages (`git log`)
- Current task progress (TaskCreate kullan)
- Ephemeral conversation state
- MEMORY.md'de zaten olan bilgiler (önce duplicate check)
- "Hatırlatma için" generic notlar (CLAUDE.md'ye yazılır, memory'ye değil)

### Yazım Formatı
```markdown
---
name: kebab-case-slug
description: tek satır relevance hook — future-Claude bu satıra bakıp ilgili mi diye karar verecek
metadata:
  type: feedback | project | reference | user
---

Kural / fact tek satır.

**Why:** Sebep — incident, hard preference, deadline.
**How to apply:** Hangi durumda, nereye, ne zaman uygulanır.

İlgili: [[other-memory-slug]]
```

Sonra `MEMORY.md`'ye **tek satır** pointer ekle:
```
- [Başlık](file.md) — bir cümle hook
```

---

## Anti-Patterns (Yapma!)

| Anti-pattern | Doğrusu |
|---|---|
| `MEMORY.md`'yi atlayıp keşfe başlamak | Önce oku — zaten yüklü |
| Smart-explore yerine reflex Glob+Grep+Read | `smart_search` tek çağrıda yapar |
| `graphify-out/GRAPH_REPORT.md` varken raw search | Önce rapor — community + god nodes hazır |
| Manuel `code-review-graph build` | Hook auto-update yapıyor, ellemey |
| Geçmiş observation'ı sadece title'ından citation | `get_observations` ile içeriği al — title yalan söyleyebilir |
| Memory yazmadan duplicate check atlamak | Önce `mem-search`, varsa **update** — yeni dosya değil |
| "Sonraki sprint" diye memory'ye ertelemek | Kullanıcı kuralı: HER ŞEYİ ŞİMDİ ÇÖZ |
| Memory'de fact claim ederken kodu doğrulamamak | Memory frozen-in-time — recommend etmeden önce `grep`/`Read` ile verify |
| Aynı bilgiyi 3 ayrı dosyaya yazmak | Tek dosya, gerekirse `[[link]]` |
| Stale memory'yi yaşatmak | Eskimiş/yanlış memory'yi sil veya güncelle |

---

## Pratik Operasyon Notları

- **Memory ≠ source of truth.** Kod, git, DB, log her zaman önceliklidir. Memory bunlara *referans* tutar.
- **Frozen-in-time:** Bir memory satırı yazıldığı anın gerçeğidir. Kullanmadan önce hâlâ geçerli mi diye state'i yokla.
- **`$CMEM` içeriği** sessiona her seferinde inject olur; token bütçeni planla — `get_observations` ile derinleşeceksen önce `search`+`timeline` ile filtrele (10x token tasarrufu).
- **Privacy:** Hassas bilgi her zaman `<private>...</private>` içine. Bu içerik claude-mem DB'sine girmiyor (tag stripping hook layer'da).
- **Worker offline?** Port 37777'de claude-mem worker çalışmıyorsa MCP araçları timeout verir. Restart: plugin reload veya `npx claude-mem` üzerinden worker yeniden başlat.
- **graphify çıktısı eskimişse** (`graphify-out/graph.json` aylar önce oluşturulduysa) → `--update` ile incremental yenile veya komple yeniden çalıştır. Eski grafa körü körüne güvenme.
- **Subagent çağrılarında**: agent'ın kendi `.claude/agent-memory/<name>/` klasöründe state biriktirdiğini unutma — context isolated ama state persistent.

---

## Hızlı Komut Referansı

```
/mem-search              → claude-mem search skill (3-layer workflow)
/make-plan               → phased plan with doc discovery
/do                      → execute phased plan
/timeline-report         → project history narrative
/knowledge-agent         → build/query corpus
/graphify [path]         → one-shot knowledge graph
/loop <interval> <cmd>   → recurring task
/schedule                → cron remote agent
```

MCP tool çağrı örnekleri (skill'ler arka planda bunları kullanır, ama direkt de erişilebilir):
```
search(query="auth bug", type="observations", limit=20, dateStart="2026-04-01")
timeline(anchor=11131, depth_before=3, depth_after=3)
get_observations(ids=[11131, 10942])
smart_search(query="shutdown", path="./src", max_results=15)
smart_outline(file_path="services/worker.ts")
smart_unfold(file_path="services/worker.ts", symbol_name="shutdown")
build_corpus(name="hooks-expertise", concepts="hooks", limit=500)
```
