# Planning Seed — FERC Interconnection Queue Resurrection (deferred)

**Status:** Planning seed. Run `/brainstorm` (or just direct fix) next session.
**Driver:** External strategy doc explicitly calls out grid interconnection queue length as a top indicator. Our FERC ingestion is broken (upstream URL gone).

## Why

The "power becomes the real bottleneck" thesis hinges on grid capacity. The FERC interconnection queue length is the canonical real-time signal of how constrained that bottleneck is. We had this ingestion in place but it crashed during the latest refresh because the Berkeley Lab URL `https://emp.lbl.gov/sites/default/files/queued_up.xlsx` started returning 404.

## Investigation needed

1. **Find the new canonical URL.** Check https://emp.lbl.gov/queues for the current queued_up.xlsx location, or use Internet Archive's most recent crawl to confirm what changed.
2. **Alternative sources** if Berkeley Lab discontinued the dataset:
   - PJM, MISO, CAISO, ERCOT each publish their own interconnection queue files (XLSX or CSV)
   - DOE EIA publishes a synthesized version (less timely)
3. **Decide:** single-URL fail-soft (current code, just update URL) vs. multi-source aggregator (more robust but more code)

## Scope sketch

**Minimal fix (recommended):**
- Update `_FERC_QUEUE_DEFAULT_URL` constant in `ingestion/ferc_queue_ingestion.py`
- Verify schema still parses (column names may have changed)
- Re-run, confirm pipeline_health goes green for FERC

**Medium fix (if URL truly gone):**
- Re-source from PJM directly (PJM is the largest queue and the strategy's main concern for the East Coast AI/DC buildout)
- Update parser for PJM's column conventions
- Test against actual file

**Full fix (if we want robust signal):**
- Multi-ISO aggregator that pulls PJM + MISO + CAISO + ERCOT
- Aggregate by state (already what _DC_STATES filtering does)
- Compute queue-length feature = total MW in queue per state, by status

## Why deferred

User flagged it as bottleneck signal worth chasing, but the immediate work today (orchestration trio + score harness + ticker expansion) was higher leverage. Picking this up next session lets us focus on the URL/parser investigation cleanly.

## Trigger

Run when: user says "fix FERC" or "do the FERC plan". Start with web check on emp.lbl.gov/queues to find new URL, then either update constant (minimal) or rewrite (full).
