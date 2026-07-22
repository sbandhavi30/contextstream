# Contributing Tool Schemas

Tool schemas teach ContextStream how to extract structured lessons from raw tool output.
Every schema you add improves extraction quality for everyone using that tool.

---

## Quickstart (5 minutes)

```bash
# 1. Scaffold a new schema interactively
python scripts/new_schema.py

# 2. Add real tool output examples to the generated test file
# bench/eval_datasets/<your_tool>_eval.json

# 3. Run eval — iterate until pass rate >= 80%
python bench/eval_extractor.py --tool <your_tool>

# 4. Validate schema structure
python scripts/validate_schemas.py

# 5. Open a PR
```

---

## Schema file format

`configs/tool_schemas/<tool_name>.yaml`

```yaml
description: "One-line description of what this tool does"
base: rest_api          # inherit fields from: bash | sql | rest_api | kubectl | file
model: claude-sonnet-4-6
fields:
  field_name: "type — description of what to extract"
  other_field: "string | null — present only if X"
```

### Field description syntax

```
field_name: "<type> — <what to extract and when>"
```

**Types:** `string`, `integer`, `float`, `list[string]`, `dict`, `boolean`  
**Nullable:** append `| null` — tells the model to return null rather than guess  
**Enum values:** list them: `'value1' | 'value2' | 'value3'`

**Good field descriptions:**
```yaml
issue_key:    "string — Jira issue key e.g. 'PROJ-123'"
rows_affected: "integer | null — row count if present in output"
status:       "string — 'open' | 'closed' | 'merged' | 'pending'"
error:        "string | null — error message verbatim if status >= 400"
confidence:   "float 0.0-1.0 — lower if output is truncated or ambiguous"
```

**Bad field descriptions (too vague):**
```yaml
info:   "information"        # what information?
result: "the result"         # useless
data:   "some data"          # model will hallucinate
```

---

## Choosing a base schema

| Your tool type | Use base |
|---|---|
| REST API / webhook response | `rest_api` |
| Database query result | `sql` |
| Kubernetes / container tool | `kubectl` |
| Config file / YAML / dotenv | `file` |
| Any CLI / shell command | `bash` |

The base schema provides common fields (status_code, error_message, etc.) so you only define what's unique to your tool.

---

## Choosing a model

| Model | Best for |
|---|---|
| `claude-haiku-4-5-20251001` | CLI output, kubectl, bash, Terraform — structured operational text |
| `claude-sonnet-4-6` | REST APIs, SQL, JSON-heavy output, nuanced error messages |

When in doubt: use `sonnet`. It's slower and costs more but handles ambiguous output better.

---

## Writing good eval cases

Each schema needs at least 3 eval cases in `bench/eval_datasets/<tool_name>_eval.json`:

```json
[
  {
    "id": "mytool_001",
    "tool": "mytool",
    "description": "What this case tests",
    "raw_output": "paste real output here — the messier the better",
    "expected": {
      "field_name": "expected extracted value",
      "root_cause": "ExactEntityName did X — metric=Y",
      "entity_targets": ["entity1", "entity2"],
      "confidence_min": 0.85
    }
  }
]
```

**Three cases minimum:**
1. **Happy path** — complete output, all fields present, `confidence_min: 0.85`
2. **Partial/ambiguous** — missing fields, no clear cause, `confidence_max: 0.75`
3. **Error case** — failed operation, error message present

**Use real output.** Copy-paste from actual tool runs. Synthetic output trains the schema wrong.

**confidence bounds:**
- `confidence_min`: model must be AT LEAST this confident (clear cases)
- `confidence_max`: model must NOT exceed this (ambiguous/vague cases — prevents overconfidence)

---

## Eval scoring

```
python bench/eval_extractor.py --tool <your_tool>
```

| Score | Meaning |
|---|---|
| `field_coverage` | Did model fill fields it should have filled? |
| `field_accuracy` | Are filled values correct? |
| `confidence_bound` | Is model confidence appropriately calibrated? |
| `root_cause_score` | Does root_cause contain the right entities and metrics? |

**Target: composite >= 0.80 on all cases before submitting PR.**

If a case fails:
- `acc` low → field description too vague — add examples or enums
- `conf_bound FAIL` (overconfident) → add calibration note to field: `"confidence: ... MUST be < 0.70 if error_message is generic"`
- `rc` low → root_cause prompt issue — check entity_targets are specific names, not generic words

---

## PR checklist

- [ ] `configs/tool_schemas/<tool_name>.yaml` — schema file
- [ ] `bench/eval_datasets/<tool_name>_eval.json` — min 3 eval cases
- [ ] `python scripts/validate_schemas.py` passes with no errors
- [ ] `python bench/eval_extractor.py --tool <tool_name>` pass rate >= 80%
- [ ] Description in YAML is one clear sentence
- [ ] All fields have type + description (not just a word)
- [ ] At least one `confidence_max` case for ambiguous output

---

## Real examples to learn from

| Schema | Notes |
|---|---|
| `configs/tool_schemas/jira.yaml` | REST API base, domain-specific status enum |
| `configs/tool_schemas/terraform.yaml` | bash base, numeric resource counts |
| `configs/tool_schemas/aws_cli.yaml` | bash base, service + operation pattern |

---

## Questions?

Open an issue with label `schema-help`. Include:
- Tool name and a sample of real output (redact secrets)
- What fields matter for your use case
