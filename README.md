# Review Sentiment Agent Team: Fine-Tuning, Validation, Tool Calling and Multi-Agent Orchestration

A small end-to-end learning project in Google Colab: fine-tune an open model, validate it on held-out data,
compare it with a GPT baseline, expose it as a tool for an agent, and orchestrate multiple agents around it.

## Goal
Classify short movie reviews as positive or negative with a fine-tuned model, then have a team of agents
classify a batch of reviews and write a short insight note.

## Data
Rotten Tomatoes movie reviews (`cornell-movie-review-data/rotten_tomatoes` on Hugging Face), labels 0 = negative, 1 = positive.

| Split | Size | How |
|---|---|---|
| Train | 1,000 | random sample, seed 42 (512 negative / 488 positive, so balanced) |
| Validation | 200 | random sample, seed 42, used during training to watch for overfitting |
| Test | 100 | random sample, seed 42, never seen in training, used once for the final score |

## Method
1. **Tokenization.** DistilBERT's tokenizer converts text to numbers (`input_ids`) plus an `attention_mask`,
   with truncation and padding to 64 tokens. Batched mapping makes preprocessing faster.
2. **Fine-tuning.** `distilbert-base-uncased` with a new 2-class classification head, trained with the Hugging Face
   `Trainer` for 3 epochs, batch size 16, default learning rate, on a free Colab T4 GPU. Training updates all of the
   model's weights, not just the new head.
3. **Validation.** Accuracy, macro-F1 and a confusion matrix on the held-out test set.
4. **Baseline.** GPT (`gpt-4o-mini`, temperature 0, one-word answers) used zero-shot on the same 100 test reviews.
5. **Tool calling.** The fine-tuned model is wrapped as a `classify_review` function tool for an agent. The agent is told
   never to decide a label itself.
6. **Multi-agent orchestration.** OpenAI Agents SDK handoffs: Triage agent -> Classifier agent (uses the tool) -> Summary agent.

## Results (n = 100 test reviews)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| DistilBERT, fine-tuned on 1,000 reviews | 0.75 | 0.75 |
| GPT-4o-mini, zero-shot | 0.90 | 0.8993 |

With only 100 test reviews, each score carries roughly +/- 8 points of sampling noise, so the gap is real in direction
but the exact numbers are rough.

**What the errors looked like.** DistilBERT's errors were balanced (12 false positives vs 13 false negatives), so it
was not biased toward one class. They clustered in negation ("not as good as it should be"), hedged opinions,
sarcasm, and a few reviews whose labels look debatable (label noise).

## Key concepts I learned
- **Fine-tuning vs prompting.** Fine-tuning changes the model's weights using labeled examples. Prompting and settings
  like temperature change only how a model is asked, not what it is. DistilBERT is a classifier, so temperature and
  reasoning settings do not apply to it.
- **New head vs pretrained body.** Loading the model prints MISSING keys (the new classification head, randomly
  initialized and learned during training) and UNEXPECTED keys (the original language-model head, discarded).
- **Epoch.** One full pass over the training data. I trained for 3 epochs.
- **Overfitting.** Training loss keeps falling while validation loss starts to rise. The validation set exists to catch this.
- **Why held-out test data matters.** Scoring on reviews the model never trained on avoids data leakage. Seeds make the
  random shuffle repeatable, so results can be reproduced.
- **Metrics.** Accuracy, precision, recall, F1 (macro-averaged across classes) and the confusion matrix. ROUGE and BLEU
  are for free-text generation and do not apply to a classification label.
- **Confidence is not accuracy.** A softmax confidence score says how sure the model is, not how often it is right.
- **Sampling noise.** Small test sets give wide error bars, so I report the sample size alongside every score.
- **Tool call vs handoff.** A tool call asks a function to do a job and returns the result to the same agent. A handoff
  transfers control to another agent, which takes over the conversation. In this project the Triage agent routes,
  the Classifier calls the tool once per review, and the Summary agent writes the final answer (confirmed with
  `result.last_agent.name`).
- **Why instruct the agent to always use the tool.** Without that instruction, a language model can skip the tool and
  answer from its own knowledge, which defeats the purpose of using the fine-tuned model.
- **Reading an agent trace.** For 3 reviews the trace had 7 items: 3 tool calls + 3 tool outputs + 1 final message.
  Count the calls, not the items.

## Challenges
- **OpenAI fine-tuning was unavailable.** My first plan was to fine-tune a GPT model through the OpenAI API. The job failed
  with a 403 `training_not_available` error because self-serve fine-tuning is closed to accounts that have not fine-tuned
  before. I pivoted to fine-tuning an open model (DistilBERT) with Hugging Face on a free Colab GPU, which also let me see
  the training mechanics directly.
- **Dataset loading error.** `load_dataset("rotten_tomatoes")` failed. Using the full name
  `cornell-movie-review-data/rotten_tomatoes` fixed it.
- **Async in Colab.** Colab already runs an event loop, so the agent runner needs `await Runner.run(...)`.

## Limitations
- Small data: 1,000 training reviews and a 100-review test set. Results are illustrative, not production-grade.
- The comparison is not like-for-like: DistilBERT is small and trained on little data; GPT-4o-mini is a much larger model.
- Binary movie-review sentiment is an easy task, so it says little about harder domain-specific problems.
- No deployment, monitoring or cost controls.

## Next steps
- Train on the full dataset (about 8.5K reviews) and evaluate on the full test set for tighter error bars.
- Tune learning rate and epochs, and try dynamic padding.
- Apply the same pipeline to a domain-specific label set, such as categorizing health-related text.
- Add guardrails and an evaluation of the agent's written summary.

## Run it
Open the notebook in Google Colab with a T4 GPU runtime. Set `OPENAI_API_KEY` as a Colab secret for the GPT baseline
and the agents. Never commit your API key, and clear notebook outputs that might display it.
