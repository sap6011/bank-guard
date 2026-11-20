# Modelling

## Summary

BankGuard's laundering classifier is a sequence transformer: a small transformer, trained from scratch, that reads each transaction together with the most recent transactions of its sender and receiver. Trained on synthetic bank transactions from 1 to 6 September and chosen on validation days 7 and 8 September, it reaches a test PR-AUC of 0.339 on 9 and 10 September (862,792 transactions, 956 of them laundering), catching 45.6% of laundering at a 0.1% false positive rate and 78.6% at 1%. Gradient boosting on the same 22 engineered features reaches 0.131 on the same test days. Eight model families were compared under rules fixed before their results, from one-line rules to fine-tuned language models and a tabular foundation model, and the sequence model led on every metric. Two limits stand out: laundering sent by hub accounts, about 12% of laundering, is missed by every model, and on 9 September a wave of extra, occasionally active accounts sharply reduces precision for all models.

## Problem and data

The task is to rank bank transfers by how likely they are to be money laundering, so that an analyst with a fixed alert budget sees as much laundering as possible. The data is the IBM Transactions for Anti Money Laundering dataset, HI-Small variant: synthetic transactions published with the NeurIPS 2023 paper "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models" (Altman et al.). It was chosen because it has sender and receiver accounts, a laundering label and named laundering typologies, so the classifier, the agents that investigate its alerts and the evaluation all describe the same problem. PaySim was rejected as too easy, and IEEE-CIS because it has no counterparties to investigate.

| Property | Value |
|---|---|
| Transactions | 5,078,345 |
| Laundering transactions | 5,177 (about 0.10%) |
| Accounts, entities, banks | 518,581 accounts, 166,207 entities, 30,470 banks |
| Date range | 1 to 18 September 2022 |
| Laundering attempts in the patterns file | 370, across 8 typologies |
| Laundering transactions in a listed pattern | 3,209 of 5,177 |

Four findings from the exploratory analysis (`01_eda.ipynb`) shaped the method. First, nearly all transactions fall on 1 to 10 September; the 11 to 18 September tail holds about 1,100 rows, more than half of them laundering, so it is left out of all headline results and the date itself is never used as a feature. Second, payment format is a shortcut: ACH carries 4,483 of the 5,177 laundering transactions, so it is kept out of the models and reported only as an ablation. Third, about 1,970 laundering transactions belong to no listed pattern, so results are broken down by group rather than by typology alone. Fourth, 591,212 transactions move money from an account to itself; they are kept and flagged, but excluded from the counts of account activity.

## How models were evaluated

**Split by time.** Models train on 1st to 6th September, are compared and chosen on 7th and 8th September (validation: 965,524 transactions, 1,036 laundering), and are scored once more on 9th and 10th September (test: 862,792 transactions, 956 laundering), about 64%, 19% and 17% of the rows. A time split matches how a bank would use the model, scoring days it has not seen, and prevents information from later transactions leaking into training. No random split is used.

**Metrics.** Laundering is about 0.1% of transactions, so accuracy and ROC-AUC say little: a model can score a high ROC-AUC while its alert list is almost all false alarms. The primary metric is PR-AUC (average precision), which rewards ranking laundering at the very top of the list. Two operational metrics go beside it: recall at a 0.1% and at a 1% false positive rate, that is, the share of laundering caught when an analyst reviews the top 0.1% or 1% of normal transactions. Recall is also broken down by group: laundering in a listed pattern, unpatterned laundering from quiet senders, and unpatterned laundering from hub senders, defined as senders with at least 1,000 payments in the previous three days.

**Rules fixed in advance.** To keep the comparison honest, each rule below was written down before the results it governs were seen, and kept even when a result landed just outside it:

1. The model choice is made on validation and locked before any test score exists. The chosen model's test score is the headline result; other models are scored on the test set for reference only and cannot change the choice.
2. Every model is compared at default or fixed settings, with no tuning on validation beyond early stopping, since LightGBM was not tuned either.
3. The text and sequence models learn from the same 202,530 training rows (all 2,530 laundering rows plus the same 200,000 normal rows), checked by matching hashes of their inputs across machines.
4. A weighted sample of validation was to be used for slow models only if it matched full validation within 0.01 PR-AUC on five random samples. The largest difference was 0.0107, so every model was scored on all of validation instead.
5. TabPFN's context size was the largest that fitted in GPU memory, chosen on memory alone and never on validation score.
6. The final combination test, an equal-weight average of the ranks of the three best models on validation with nothing else tried, ran only after every model was finished.

## Features

Every model reads the same 22 features, built once by `bankguard.features` from the full transaction table. Each count uses only transactions strictly earlier than the row being scored, a rule enforced by tests, so no feature can see the future.

| Group | Features | What they capture |
|---|---|---|
| Transaction (5) | log amount, amount relative to the training median for its currency, hour of day (as sine and cosine), self transfer flag | The transfer itself |
| Account activity (9) | payments sent and received by the sender and by the receiver over the previous 1 and 3 days, and earlier transfers between the same pair in the previous 3 days | How busy each account is, and whether the two have dealt before |
| Counterparty breadth (2) | recent payments by the sender to new counterparties, and by the receiver from new counterparties | Fanning out or in to unfamiliar accounts |
| Owner attributes (6) | entity type and number of accounts held by the owner, for each side; whether both sides belong to the same owner or the same bank | Who holds the accounts |

The ladder of LightGBM models in `02_baselines.ipynb` shows where the signal is. On validation, the transaction features alone reach PR-AUC 0.007, no better than flagging every ACH payment. Adding account activity raises it to 0.270, about forty times higher; counterparty breadth adds a little (0.302) and owner attributes more (0.341, ranging from 0.331 to 0.342 across seeds). Laundering in this data is visible in how accounts behave across transactions, not in single transfers.

Three strong signals are kept out of the models because they appear to come from how the simulator generates data rather than from laundering itself. Payment format and currency would lift the account activity model from 0.270 to 0.446, mostly because ACH carries 87% of laundering. Bank country would lift the full model from 0.341 to 0.450. And the count of earlier transfers between the same pair, which is kept, carries much of the signal: without it the account activity model falls to 0.104, because only 4.6% of validation transfers go to a counterparty not paid in the previous three days, yet they contain 842 of the 1,036 laundering transactions; real customers repeat counterparties far less reliably than the simulator does.

## Models tried

Models were added one family at a time, each in its own notebook, all on the same features, split and metrics.

| Notebook | Model | Idea |
|---|---|---|
| `02_baselines` | Two one-line rules, logistic regression, LightGBM | Floors to beat, then gradient boosting on the 22 features |
| `03a_bert_finetune`, `03b` | DistilBERT, ModernBERT (2 and 4 epochs) | Write each row's features as a sentence and fine-tune a pretrained text encoder |
| `04a_tabpfn`, `04b` | TabPFN-3.5-Fast | A tabular foundation model that predicts from 200,000 labelled rows given as context, with no training on our data |
| `05a_qwen_lora`, `05b`, `05c` | Qwen3-0.6B with LoRA | A small decoder language model, fine-tuned on the same sentences by training 0.76% of its weights |
| `06a_sequence_model` | Sequence transformer | Reads the current row's features plus the last 64 transactions of the sender and the receiver |

Results on validation (965,524 transactions, 1,036 laundering):

| Model | PR-AUC | Recall at 0.1% FPR | Recall at 1% FPR |
|---|---|---|---|
| Rule: flag every ACH payment | 0.007 | 0.000 | 0.000 |
| Rule: flag every new counterparty | 0.016 | 0.000 | 0.000 |
| DistilBERT | 0.228 | 0.296 | 0.726 |
| ModernBERT, 2 epochs | 0.302 | 0.360 | 0.783 |
| Qwen3-0.6B with LoRA | 0.306 | 0.357 | 0.764 |
| LightGBM | 0.341 | 0.379 | 0.764 |
| ModernBERT, 4 epochs | 0.353 | 0.375 | 0.761 |
| TabPFN-3.5-Fast | 0.398 | 0.397 | 0.796 |
| Sequence transformer, single seed (mean of 3) | 0.457 | 0.518 | 0.826 |
| **Sequence transformer, average of 3 seeds** | **0.486** | **0.527** | **0.839** |

The rules cannot rank, only flag, and each flags far more than 1% of normal transactions, so they fit neither alert budget; logistic regression on the transaction features alone reaches 0.002. Every model that reads the features as text sits at or below LightGBM, even Qwen at four times the size of ModernBERT: pretraining on language adds little when the input is 22 numbers. TabPFN-3.5-Fast beats LightGBM clearly without any training on our data, but its weights are licensed for non-commercial use only. The sequence transformer leads on every metric, and even a single seed is well ahead of TabPFN. The average of three seeds was chosen as its entry before its results were seen; averaging adds about 0.03 PR-AUC over a single seed.

## Error analysis

Overall scores hide which laundering is caught. Recall at a 1% false positive rate on validation, by group:

| Group | Laundering | LightGBM | DistilBERT | ModernBERT, 2 ep. | ModernBERT, 4 ep. | Qwen | TabPFN | Sequence |
|---|---|---|---|---|---|---|---|---|
| In a listed pattern | 657 | 0.913 | 0.854 | 0.930 | 0.896 | 0.901 | 0.945 | **0.989** |
| Unpatterned, quiet sender | 246 | 0.764 | 0.772 | 0.813 | 0.793 | 0.809 | 0.829 | **0.890** |
| Unpatterned, hub sender | 133 | 0.023 | 0.008 | 0.000 | 0.030 | 0.000 | 0.000 | 0.000 |

The sequence transformer leads on both groups that can be detected, and its largest gain is on unpatterned laundering from quiet senders: 89.0% against 76.4% for LightGBM, the laundering that matches no known typology and is the hardest to write rules for.

Laundering sent by hub accounts, senders with at least 1,000 payments in the previous three days, is missed by every model: between 0% and 3% recall across gradient boosting, three fine-tuned language models, a tabular foundation model and a sequence model. These transactions go to known counterparties at normal amounts, so neither the 22 features nor an account's last 64 transactions separate them from the hub's legitimate traffic; for a hub, 64 transactions cover only a few minutes. With hub sent laundering undetected, recall cannot exceed (657 + 246) / 1,036 = 87.2%. The sequence transformer reaches 83.9%, about 96% of what the inputs allow, so further gains need new information about hubs, such as graph structure across many accounts, rather than a better model on the same inputs.

By length of the sender's history, the sequence transformer's gain comes from accounts with 1 to 63 earlier transactions, where it catches 93.7% to 97.8% of laundering against 82.1% to 93.1% for the other models. Rows with a full 64-transaction history stay at about 40% for every model, because 133 of their 225 laundering rows come from hub senders.

## The sequence transformer

The 22 features compress an account's past into a few counts. The sequence transformer reads the past itself: for each transaction, the most recent transactions of its sender and of its receiver, alongside the current row's 22 features, so any gain over LightGBM comes from the history. Its design was fixed in the PRD before any results.

**Histories:** Each transaction is an event for its sender and one for its receiver, and a history holds an account's last 64 events strictly before the current minute. Each event carries six fields: log amount, amount relative to its currency median, sent or received, hours before the current transaction, self transfer flag, and whether it involves the same counterparty as the current transaction. Laundering labels of past transactions are never used, since a bank learns them only weeks later. The length of 64 was set from the data before training: the median validation transaction has 59 earlier transactions on the sender side and 33 on the receiver side (29 and 24 for laundering), so 64 covers the full history for about three quarters of laundering rows. `bankguard.sequence` finds each history by binary search over sorted events instead of storing sequences, and a randomised test checks 2,000 histories for any later event.

**Model and training:** Each side's history passes through its own two-layer transformer (four attention heads, width 64) and is summarised by a learned summary token; the two summaries and the current row's features feed a small classifier, 224,065 parameters in all. It trains from scratch on the laptop GPU on the same 202,530 rows as the text models, with AdamW at learning rate 0.001, batches of 256 and dropout 0.1, keeping the epoch with the best PR-AUC on a fixed fifth of validation and stopping after three epochs without improvement. Three seeds reach validation PR-AUC 0.454, 0.482 and 0.434, a wider spread than LightGBM's, and their average reaches 0.486.

**Time shift check:** The data starts on 1 September, so early training rows can look back only a day or two: a full 64-event sender history exists for 11% of training rows against 36% of validation rows. As planned before the main run, the sequence transformer and LightGBM were both retrained on the 86,816 training rows dated 3 to 6 September, which hold 1,800 of the 2,530 laundering rows. On these rows the sequence transformer reaches 0.514 against 0.348 for LightGBM, a gap close to the one on all 202,530 rows (0.486 against 0.286), so its lead is not an artifact of short training histories. Both models improve when the first two days are dropped, LightGBM more, which suggests rows with histories cut short by the start of the data mislead training; a future version would use the first days as history only. This check was an ablation and did not affect model selection.

**Limitations:** The "same counterparty" field is related to the pair count, which is partly a simulator signal, so part of the gain may not transfer to real data. Earlier transactions of the same laundering scheme can appear in a history; this is legitimate, since only earlier transactions are used, but it makes structured typologies easier to recognise here than they may be in practice.

## Model choice

As fixed in advance, the three models with the highest validation PR-AUC were combined by an equal-weight average of their ranks, with no other combination tried, and the better of the combination and the best single model was chosen.

| Model | PR-AUC | Recall at 0.1% FPR | Recall at 1% FPR |
|---|---|---|---|
| Rank average of the sequence transformer, TabPFN-3.5-Fast and ModernBERT (4 epochs) | 0.452 | 0.465 | 0.805 |
| **Sequence transformer, average of 3 seeds** | **0.486** | **0.527** | **0.839** |

The combination scores below the sequence transformer alone on every metric and on both detectable groups (95.3% against 98.9% on patterned laundering, 84.1% against 89.0% on quiet senders). A combination helps when models make different mistakes; here the other two mostly catch a subset of what the sequence transformer already catches, and rank it less sharply, so an equal vote dilutes the stronger ranking. A weighted combination might do better but would mean tuning on validation, which the rules exclude. The sequence transformer, as the average of its three seeds, was therefore chosen and locked before any test score was computed.

Validation was used for early stopping, for choosing between the two ModernBERT runs and for this choice, so all validation scores above are somewhat optimistic. The test set gives the unbiased figure.

## Test results

The test days, 9 and 10 September, hold 862,792 transactions, 956 of them laundering. The sequence transformer's score is the headline result; the other models are reported for reference only and could not change the choice. Test scores for TabPFN-3.5-Fast, ModernBERT and Qwen are to be added.

| Model | Validation PR-AUC | Test PR-AUC | Test recall at 0.1% FPR | Test recall at 1% FPR |
|---|---|---|---|---|
| **Sequence transformer, average of 3 seeds (chosen)** | 0.486 | **0.339** | **0.456** | **0.786** |
| LightGBM | 0.341 | 0.131 | 0.186 | 0.516 |
| DistilBERT | 0.228 | 0.056 | 0.093 | 0.346 |

By group at a 1% false positive rate, the sequence transformer catches 96.4% of patterned laundering (585 transactions) and 71.6% of unpatterned laundering from quiet senders (261), against 67.0% and 38.7% for LightGBM. As on validation, none of the 110 laundering transactions from hub senders is caught by any model.

The ranking holds and the sequence transformer's lead widens, from 1.4 times LightGBM's PR-AUC on validation to 2.6 times on test. But every model scores far lower on test than on validation, and unevenly: the sequence transformer by 30%, LightGBM by 62% and DistilBERT by 75%. A drop of this size points to a difference in the data rather than optimism from model selection, and the scores by day locate it:

| Day | Split | Transactions | Laundering share | Sequence | LightGBM | DistilBERT |
|---|---|---|---|---|---|---|
| 7 Sep | validation | 482,751 | 0.103% | 0.494 | 0.354 | 0.247 |
| 8 Sep | validation | 482,773 | 0.112% | 0.478 | 0.330 | 0.211 |
| 9 Sep | test | 654,467 | 0.079% | 0.262 | 0.071 | 0.037 |
| 10 Sep | test | 208,325 | 0.212% | 0.477 | 0.389 | 0.203 |

On 10 September every model scores about as on validation; the drop is confined to 9 September. That day holds about 36% more normal transactions than any validation day, spread evenly across every hour, from 205,570 distinct senders against about 120,450 on a normal day, while the payment format mix and the share of self transfers barely change. A lower laundering share alone would lower PR-AUC by about a quarter, not fivefold as for LightGBM, so the extra transactions must be normal ones that the models score as suspicious: a wave of occasionally active accounts, whose short histories and new counterparties resemble laundering behaviour. 10 September, with fewer transactions and senders and a higher laundering share, matches the thinning of the data before the 11 to 18 September tail.

This is a shift in the data at the end of the simulation, not a failure to generalise to later days, and the sequence transformer, which reads each account's actual history instead of trailing counts, degrades least: on 9 September it keeps 3.7 times LightGBM's PR-AUC. It is also the kind of drift a deployed model must be watched for, so the planned drift check monitors the number of active accounts and the distribution of scores, not only feature averages.

## Limitations and next steps

**Limitations**
- **Synthetic data.** All results come from a simulator. Several strong signals (payment format, the pair count, bank country) appear to be properties of the simulator rather than of laundering; the strongest are kept out of the models and the rest are reported, but part of the measured performance would not transfer to a real bank.
- **Hub sent laundering is not detected.** About 12% of laundering is invisible to every model, which caps recall at about 87%.
- **Shift on the test days.** One test day carries a wave of extra accounts that sharply lowers precision for every model, and the headline test score includes it.
- **Stability.** The sequence transformer varies more between seeds than LightGBM (validation PR-AUC 0.434 to 0.482); the text models and TabPFN were run once each.
- **Not yet done.** The PRD plans calibration, a cost based decision threshold and a comparison with the results reported in the dataset paper; these are not covered here.

**Next steps**
- **Graph structure for hub senders.** A graph neural network over the transaction network, planned for Phase 2, is the natural next model for the group no current input separates.
- **Training data.** Use the first days of data as history only, and train the sequence transformer on more normal rows; LightGBM gains clearly from its full 3.25 million rows.
- **Serving.** The sequence transformer is small enough for a CPU, but it needs each account's recent transactions at scoring time, which the serving design must provide.
- **Monitoring.** Track the number of active accounts and the score distribution, so that a shift like 9 September is caught as it happens.
- **Stretch goals.** TabPFN-2, the version licensed for commercial use, and test scores for the remaining models.

## How to reproduce

1. Install the project with `uv sync`. Python 3.13 is pinned in `.python-version`.
2. Download the IBM Transactions for Anti Money Laundering dataset from Kaggle (`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`) and place `HI-Small_Trans.csv`, `HI-Small_accounts.csv` and `HI-Small_Patterns.txt` in `data/`.
3. Run the notebooks in order: `01_eda`, `02_baselines`, `03a_bert_finetune`, `04a_tabpfn`, `05a_qwen_lora`, `06a_sequence_model`, then `model_comparison`. Notebooks ending in `_kaggle` record the exact runs on Kaggle GPUs and need a Kaggle session with two T4 GPUs to rerun.
4. Every run is logged to MLflow in `mlflow.db`; browse them with `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.
5. Run the tests with `uv run pytest`. Saved models and scores live in `models/`, which is not committed, so reproducing the numbers means rerunning the notebooks.

