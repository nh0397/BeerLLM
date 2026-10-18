# Evaluate actual brewing quality

The automatic report compares the exact same held-out prompts with the base model and with LoRA enabled. Outputs are deterministic (greedy decoding). The default sample is three recipes to keep CPU runtime manageable; increase `--limit` for a broader final assessment. Use the validation split while tuning, then reserve the test split for the final comparison.

The report includes assistant-token loss against generated reference guides, ingredient-name recall, stage mentions and uncertainty wording. These are diagnostic proxies, not accuracy scores: a wrong quantity alongside the right ingredient still counts as a name match. Repeated recipe phrasing can lower loss without improving brewing competence. Read the actual answers.

For each base/adapter pair, review:

| Check | Pass condition |
|---|---|
| Ingredients | No omitted or invented ingredients; correct quantities, units and batch size |
| Mash | All source temperatures/durations preserved in order |
| Hops | Each quantity paired with its correct addition; vague source labels are not silently converted to minutes |
| Fermentation | Correct yeast and temperature; no invented duration or claim that bubbling alone proves completion |
| Procedure | Coherent ordered instructions; specialized steps preserved where applicable |
| Packaging | Does not invent priming amounts or declare fermentation complete without evidence |
| Attribution | Distinguishes original facts, general guidance and unresolved details |
| Completeness | Acknowledges remaining gaps; incomplete records are not presented as fully verified brew sheets |

Use 0 (incorrect), 1 (partly correct), or 2 (correct) for each category and record the specific error. Have an experienced brewer review before relying on the output. Record whether generation hit the token limit; a cut-off answer needs a separate completeness check.

Named recipes seen during training can be tested separately for memorization. Do not label a supplied-context test as closed-book recall, or expect accurate recall of recipes excluded from training. Our family grouping is conservative but does not catch every alias or near duplicate; inspect splits before making benchmark claims.
