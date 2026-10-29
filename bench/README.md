# Evaluation sets

| File | What | Used by |
|---|---|---|
| `tools.jsonl` | 80 requests and the tool call each should make (or none). The system prompt and the tool descriptions were tuned on these. | `courier eval tools` |
| `tools-heldout.jsonl` | 40 more, written after that tuning and never used to change anything. | `courier eval tools --set bench/tools-heldout.jsonl` |
| `endpoint.jsonl` | 30 spoken requests, 15 with a pause in the middle (`|` marks it). | `courier eval endpoint` |
| `latency.jsonl` | 20 requests spoken after "Hey Courier" by voices the wake word never trained on. | `courier eval latency` |

Every tool case is judged at a fixed moment (Thursday 29 October 2026, 14:00, in Bucharest), so
"tomorrow" is always 2026-10-30.

## Expected calls

`expect` lists the acceptable calls; `[]` means no tool should be called. Each expected argument is
one of these:

- **A plain value.** It must be present and match:
  - numbers must be equal;
  - `date` and `time` compare as dates and times ("friday" = "2026-10-30", "6 pm" = "18:00");
  - other strings must contain the expected words, in any order and case.
- **`{"optional": v}`.** May be absent; if present, it must match `v`. `true` accepts anything.
- **`{"date": d, "optional": true}`.** Like `optional`, compared as a date.
- **`{"absent": true}`.** Must not be there.
- **`{"absent_or": v}`.** Absent, or matching `v`.
- **`{"min": a, "max": b}`.** A number in range.
- **`{"generic": true}`.** Absent or only filler words ("some music").
- **`"_result": {"seconds": 600}`.** Checks the number the tool returned rather than the words the
  model wrote. For example, "ten minutes", "10 minutes" and "600 seconds" all give a 600-second timer.
