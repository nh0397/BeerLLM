# What the evaluation metrics mean

We compare the same base model before training and with the new LoRA adapter afterward. Prompts, references, tokenizer, maximum answer lengths and greedy decoding stay fixed. Results are separated by task: four seen-recipe recall questions, three held-out supplied-recipe questions, one private-recipe uncertainty question and one general fermentation check. Nine questions are a diagnostic sample, not a representative brewing benchmark.

## Cross-entropy loss / negative log-likelihood (NLL)

This measures how much probability the model assigns to the correct next token in a reference answer. Smaller is better at predicting that reference text. Only assistant answer tokens count; the question and supplied recipe facts are masked. We multiply each example's mean loss by its supervised token count, sum those losses and divide by the total supervised tokens. A long answer therefore receives weight proportional to its token count.

The model sees the earlier correct reference tokens during this measurement. This is called teacher forcing. Consequently, good reference loss does not prove the model can generate the whole answer correctly on its own. Two equally good explanations can also use different wording and receive different losses.

## Conditional answer perplexity (PPL)

`perplexity = exp(answer-token cross-entropy)`.

For example, a loss of 2.0 gives a perplexity around 7.39; a loss of 1.0 gives about 2.72. Lower means the reference answer is more predictable to the model. It is not a percentage accuracy and does not mean a 2.72% error rate. Our result is conditional on each question and any supplied recipe facts, rather than a general corpus perplexity. Compare it only with the same tokenizer, inputs and scoring setup. This follows the [Transformers explanation of perplexity](https://huggingface.co/docs/transformers/v4.46.3/perplexity).

## Fact-pattern check pass rate

Each question has small, predefined text checks, such as a particular hop dose, fermentation temperature or timing relationship. If 12 of 16 checks match, the pattern pass rate is 75%. The report stores each matched span so errors are visible.

These checks are reproducible diagnostics, not factual accuracy. They can miss a valid paraphrase, accept a correct-looking number in the wrong explanation, or fail to detect a negation or an invented extra step. A wrong answer can pass some checks. Read the actual answers and score correctness separately. We deliberately avoid presenting ingredient-name recall or stage-keyword coverage as brewing accuracy.

## Factual correctness, groundedness and completeness

These require judging the generated answer against its source, not just measuring likelihood:

| Metric | Question it answers | Example |
|---|---|---|
| Factual correctness | Are the ingredients, amounts, temperatures and timings correct and attached to the right steps? | Does 62°F belong to fermentation or to some unrelated stage? |
| Groundedness / faithfulness | Are factual claims supported by the available recipe or identified general guidance? | Does the model invent a priming dose when the source gives none? |
| Hallucination rate | What fraction of factual claims are unsupported or contradicted? | Two unsupported claims out of ten assessed claims gives 20%, provided the claim-counting method is stated. |
| Procedure completeness | Does the answer include all required source steps and acknowledge gaps? | A guide with correct mash numbers but no fermentation stage is incomplete. |
| Instruction following | Does it answer the question in the requested form? | An ordered brewing guide should not be replaced by raw JSON. |
| Clarity | Can a beginner understand what to do and why? | Explain that “pitch” means adding yeast. |

For the small run, read every before/after answer and use a 0/1/2 rubric: 0 incorrect or missing, 1 partly correct, 2 correct for the specified criterion. Record the errors, reviewer identity and whether the answer was cut off. An assistant review is not an independent human or expert brewer review. Do not claim a measured hallucination percentage without defining and counting the assessed claims.

## Operational diagnostics

- **Token-limit hit rate:** answers that reach the maximum generation length divided by total answers. This signals possible truncation, not certain incompleteness. An answer may finish badly before the limit too.
- **Repeated four-gram fraction:** repeated occurrences of four-word sequences divided by all four-word sequences. Lower can indicate less looping, but legitimate repeated ingredient descriptions also count.
- **Generation latency:** seconds to produce the answer. Compare with token counts and similar workload; a shorter but incomplete answer can be faster.
- **Training loss versus validation loss:** training loss measures fit to training examples; validation loss measures reference prediction on separate recipe families. Falling training loss with worsening validation loss can indicate overfitting. Final test questions are kept out of training and used for the final comparison.

## Why not a single accuracy score, BLEU or ROUGE?

An answer is not one class label, so classification accuracy needs a clearly defined factual test first. Precision, recall and F1 can be useful for a defined extraction task, such as predicting a set of ingredient–quantity pairs:

- **Precision:** correct predicted pairs divided by all predicted pairs. This asks how much of what the model supplied was correct.
- **Recall:** correct predicted pairs divided by all required source pairs. This asks how much of the required information it recovered.
- **F1:** `2 × precision × recall / (precision + recall)`, balancing the two. If precision and recall are both zero, define F1 as zero.

For example, suppose there are three required pairs, the model predicts four pairs, and two are correct. Precision is 2/4 = 50%, recall is 2/3 = 66.7%, and F1 is about 57.1%. Matching must account for ingredient identity, amount and compatible units, and handle duplicates consistently. We have not implemented that extraction evaluator here, so no precision/recall/F1 result is claimed. Ingredient-name matches alone cannot establish recipe correctness.

BLEU and ROUGE measure forms of overlap with reference wording. They may be useful for some tasks, but a brewing answer can keep most words while changing a crucial temperature. For this experiment, source correctness, useful explanation and admitted gaps matter more than matching phrasing, so BLEU/ROUGE are not reported.

Successful fine-tuning requires better actual answers. Lower perplexity alone does not establish that outcome.
