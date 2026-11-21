# BankGuard: Product Requirements Document

**Status:** Draft v5, 19 Nov 2026. Supersedes draft v4. Updates the build status (packaging and CI done, step 4 under way), adds the evaluation rules fixed in advance, the notebook layout, and the TabPFN licence limit. Measured results are kept in the notebooks, with only the results that drive design decisions here. Draft v4 added a public live demo, a revised build order and the first modelling results. Draft v3.1 added a parallel fraud scorer (FRAML) to the Phase 2 roadmap. Draft v3 moved the dataset from PaySim to the IBM AML dataset (HI-Small).
**Matches:** Architecture v3 (zero idle cost), with the public demo deployment described in section 9. The v2 architecture diagram is out of date (it still shows fraud detection, Kinesis, EKS and OpenSearch) and is to be replaced by a v3 diagram.
**Naming:** "Phase 1" is the first complete build described here. "Phase 2" is the upgrade planned about six months later (section 13). Diagram versions (v1, v2, v3) are a separate numbering.

Items marked **[confirm]** are assumptions Sap should confirm or override.

## 1. Brief

BankGuard is a portfolio project that shows applied machine learning and AI engineering through a bank financial crime detection system focused on money laundering. It combines a rigorously evaluated laundering classifier with a LangGraph multi-agent investigation workflow that has human review built in. It targets data scientist, ML engineer, and AI engineer roles. Success means a reviewer can see, in under five minutes, evidence of sound modelling judgment, measurable agent quality, and the ability to ship the result. The project lives in a public GitHub repo with a live demo link and a recorded three minute demo. The live demo runs on free or scale to zero hosting and defaults to a replay mode, so public visits cost nothing. AWS services are used on demand and paid from account credits. The design also keeps a later commercial path open (section 16).

## 2. Audience and reviewer journey

**Primary reviewers:** hiring managers and interviewers for data scientist, ML engineer, and AI engineer roles, including intern and graduate tracks.

| Role | What they read first | Where BankGuard answers it |
|---|---|---|
| Data scientist | Problem framing, baselines, evaluation rigor, error analysis | Section 5 |
| ML engineer | Training pipeline, experiment tracking, reproducibility, serving, tests | Sections 5, 8 |
| AI engineer | Agent design, tool use, LLM evaluation, guardrails | Sections 4, 6 |

**What turns them off:**
- A model with one headline metric and no baseline
- Agents that are plain functions with an LLM label
- Infrastructure with no interesting code behind it
- A demo that does not match the repo

**Reviewer journey:**
1. README: one paragraph pitch, the live demo link, architecture diagram, demo video link, and a results table (model comparison and agent evaluation scores).
2. Live demo or demo video: a flagged transaction moves through investigation to analyst approval.
3. Deeper read: `docs/MODELLING.md`, `docs/AGENT_EVAL.md`, `docs/ARCHITECTURE.md`, and this PRD.
4. The hook: every claim in the README is backed by a number and a notebook or test that reproduces it.

## 3. Functional scope

End to end, the system does the following:

1. **Ingest** synthetic transaction events from a replay script through a queue.
2. **Normalise and enrich** each event (deterministic code).
3. **Score** laundering risk with the served model from the modelling track (deterministic model call).
4. **Route** on the score: low risk ends the run, high risk goes to investigation.
5. **Investigate** with an LLM agent that chooses its own tools: sanctions lookup, account history query, and similar past cases from vector memory.
6. **Loop** when evidence is insufficient: the supervisor sends the case back for more evidence, up to a fixed limit.
7. **Summarise** the case into a draft analyst report with guardrails applied.
8. **Pause** for human review. The graph checkpoints and the case appears in the dashboard.
9. **Resume** when the analyst approves or rejects. The decision is stored.
10. **Monitor** score drift on a schedule.

**Happy path for the demo:** a replayed stream produces three flagged cases. One is a large transfer to a watchlisted counterparty. The investigation agent calls two tools, the summary agent drafts the report, the analyst approves, and the case closes.

## 4. Multi-agent design

The multi-agent system is the core of the project. It takes about 30% of the effort.

| Node | Uses an LLM | Type | Responsibility |
|---|---|---|---|
| Tx ingestion | No | Deterministic node | Parse, validate, enrich |
| Anomaly scoring | No | Deterministic node | Call the classifier, apply the threshold |
| Investigation and regulatory flagging | Yes | LLM agent with tools | Gather evidence, apply AML and sanctions rules, decide whether to flag |
| Case summary | Yes | LLM agent | Draft the analyst report from case facts |
| Supervisor | No (rule based) | Router | Conditional routing, loop control, human interrupt |

**Design decisions:**
- Only two nodes are LLM agents. Parsing and scoring do not need an LLM, and using one there would add cost and nondeterminism for no gain.
- The supervisor routes on explicit conditions (score, evidence status, loop count). An LLM router would add nothing to a flow this structured.
- Real autonomy sits in the investigation agent, which decides which tools to call and when it has enough evidence.
- Semantic Kernel provides the memory abstraction and the shared tool plugins. LangGraph owns control flow.
- The modelling results shape the agent design. About 13% of laundering in validation is sent by hub accounts and no model tried so far ranks it (section 5). How such cases reach investigation, for example through rules or sampling, is an open design question for this stage.

**Capabilities demonstrated:** conditional routing, tool use, a bounded loop, vector memory, checkpointing, and a human interrupt with resume.

## 5. Modelling track

About 30% of the effort. This is the main evidence for data scientist and ML engineer roles.

**Dataset:** IBM Transactions for Anti Money Laundering, HI-Small variant. Synthetic data published with the NeurIPS 2023 paper "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models" (Altman et al.). Licence: Community Data License Agreement, exact version to be noted in the README. Three files are used: transactions, accounts, and laundering patterns.

**Why this dataset:** it has sender and receiver accounts, a laundering label, and named laundering typologies, so the model, the agent tools, and the evaluation set all describe the same problem. PaySim was rejected as too easy. IEEE-CIS was rejected as the primary because it has no counterparties for the agents to investigate.

**Verified from the downloaded files on 12 Nov 2026:**

| Property | Value |
|---|---|
| Transactions | 5,078,345 |
| Laundering transactions | 5,177 (about 0.10%) |
| Date range | 1 to 18 Sep 2022 |
| Accounts, entities, banks | 518,581 accounts, 166,207 entities, 30,470 banks |
| Laundering attempts in the patterns file | 370, across 8 typologies |
| Laundering transactions covered by a pattern | 3,209 of 5,177 |

**Findings that shape the method:**
1. **The last eight days are an artifact.** 1 to 10 Sep hold nearly all rows. 11 to 18 Sep hold about 1,100 rows, more than half of them laundering. A naive time split would make the date itself a giveaway.
2. **Payment format is a shortcut.** ACH carries 4,483 of the 5,177 laundering rows. Wire and Reinvestment carry none.
3. **Patterns cover only part of the positives.** About 1,970 laundering rows belong to no listed pattern.
4. **Self transfers are common.** 591,212 rows move money from an account to itself. They are kept, flagged, and excluded from the graph derived counts.
5. **Two columns are both named `Account`.** They are renamed to sender and receiver accounts on load.

**Method:**
1. **Exploratory analysis** with a written summary of class balance, time structure, payment formats, currencies, and leakage risks.
2. **Time based split within 1 to 10 Sep.** Train on 1 to 6 Sep, validate on 7 to 8 Sep (965,524 transfers, 1,036 laundering), test on 9 to 10 Sep (about 64%, 19%, and 17% of rows). The 11 to 18 Sep tail is excluded from headline metrics and reported separately. No random split. The raw date is never a feature.
3. **Features (22 in total):**
   - Transaction level: log amount, amount relative to the training median for its currency, hour of day, and a self transfer flag.
   - Graph derived, computed only from earlier timestamps and excluding self transfers: incoming and outgoing counts for sender and receiver over trailing 1 and 3 day windows, and prior transfers between the same pair.
   - Counterparty breadth: recent transfers to and from new counterparties.
   - Account attributes from the accounts file: entity type, accounts per entity, same owner and same bank flags, and bank country.
4. **Baseline ladder and alternative models:** two one line rules (ACH, new counterparty), logistic regression and LightGBM in `02_baselines.ipynb`. Then alternative models against LightGBM, each in its own notebook: DistilBERT and ModernBERT fine-tuned on the same features written as text, TabPFN-3.5-Fast (a tabular foundation model), Qwen3-0.6B fine-tuned with LoRA, and a sequence transformer over each account's transaction history. TabPFN-2 was planned as the commercially usable TabPFN but moved to the stretch goals on 20 Nov 2026, before any TabPFN-2 result, because the sequence model, which has its own weights, had already answered the question it was meant to answer. All models are compared on one table, overall and by laundering group, in `model_comparison.ipynb`.
5. **Imbalance handling and stability:** logistic regression uses balanced class weights. LightGBM is trained unweighted with L2 regularisation and early stopping on validation PR-AUC, after a fixed length run without them proved unstable. Headline models are retrained across seeds. The text models train on the same 202,530 sentences (all training laundering plus 200,000 sampled normal rows), checked by matching hashes across machines.
6. **Metrics:** PR-AUC as the primary metric, recall at 0.1% and 1% false positive rates, calibration, and a cost based decision threshold. ROC-AUC is reported but not used for selection, because it is misleading at this class balance.
7. **Ablation:** signals that may depend on how the data was generated are reported separately: payment format, payment currency, the pair count, and bank country.
8. **Experiment tracking:** MLflow (local SQLite store) for every run, with parameters, metrics, tags and artifacts. Runs trained on Kaggle are logged from their saved metrics files.
9. **Error analysis:** recall by laundering typology, and separately for patterned laundering, unpatterned laundering from quiet senders, and unpatterned laundering from hub senders.
10. **Model selection:** chosen on validation by the rules below; the sequence model (average of three seeds) was chosen on 20 Nov 2026 after the combination test. The model with the best research result and the model served in the live demo may differ (licence and serving cost), and `docs/MODELLING.md` says which and why. If a model loses to gradient boosting, the README says so and explains why.

**Evaluation rules fixed in advance.** Each was set before the numbers it governs were seen, and is kept even when a result lands just outside it.
1. The model choice is locked on validation before any test score exists. The chosen model's test score is the headline result. The other models are also scored on the test set for reporting only, to show whether the validation ranking holds, and their test scores never change the choice (amended 20 Nov 2026, after the choice and before any test score).
2. Weighted sampling of validation is trusted only if, over 5 random samples, every model's weighted PR-AUC is within 0.01 of its full value. The check gave a largest difference of 0.0107, so slow models are scored on all of validation, split across separate runs where needed.
3. TabPFN's context size is the largest that fits in GPU memory, chosen on memory alone and never on validation score.
4. Models are compared at default or fixed settings. None is tuned on validation beyond early stopping, since LightGBM was not tuned either.
5. The combination test (equal-weight average of the ranks of the three models with the highest validation PR-AUC, nothing else tried) runs only after every model is finished. Model choice and the single test set score follow it.

**Notebook layout.** Each model family has its own number: 03 for the BERT models, 04 for tabular foundation models, 05 for the LLM with LoRA, and 06 for the sequence model. The comparison of all models (`model_comparison.ipynb`) comes last and is numbered once the model list is final. Within a family, the "a" notebook runs on the laptop and tells the story (why the model, what was tried, results loaded from Kaggle, MLflow logging, findings), and later letters hold the exact Kaggle runs with their outputs, for example `04a_tabpfn.ipynb` and `04b_tabpfn_kaggle.ipynb`, or `05b_qwen_lora_train_kaggle.ipynb` and `05c_qwen_lora_score_kaggle.ipynb` for training and scoring. Each code cell is followed by a short written paragraph on choices and findings.

**Sequence model design, fixed on 19 Nov 2026 before any results.** Its purpose is to test whether recent transaction history carries signal the 22 features miss, in particular for hub sent laundering.
1. **Input:** the current row's 22 features plus the most recent transactions of the sender and of the receiver, so any gain over LightGBM comes from the history. The history length is set from the observed distribution of past transactions per account, before training.
2. **Each past transaction:** log amount, amount relative to its currency median, sent or received, hours before the current transaction, self transfer flag, and whether it involves the same counterparty as the current transaction. Payment format, currency and bank country are excluded, as for the other models.
3. **Leakage:** only transactions with an earlier timestamp, and never their laundering labels. A test in `src/bankguard` checks that no later transaction enters a sequence.
4. **Model:** a small transformer trained from scratch (2 layers, 4 heads, width 64, one summary token per sequence), joined with the current row's features before the classifier. Trained on the laptop GPU.
5. **Training rows:** the same 202,530 rows as the text models (all training laundering plus the same 200,000 normal rows). Every row still sees its full history.
6. **Settings:** AdamW, learning rate 1e-3, batch 256, dropout 0.1, at most 20 epochs, keeping the epoch with the best PR-AUC on the same validation subset the BERT runs used and stopping after 3 epochs without improvement. Three seeds (42, 7 and 123), reported as mean and standard deviation.
7. **History length and time shift:** a history length of 64 for both sides, set from the observed distribution (it covers the full history for about three quarters of laundering rows). Because the data starts on 1 Sep, training rows have shorter histories than validation rows (a full 64-event sender history for 11% of training rows against 36% of validation rows). The main run keeps the same 202,530 rows anyway, and results are reported by history length.
8. **Time shift check, after the main run:** the sequence model and LightGBM are both trained on the subset of those 202,530 rows dated 3 to 6 Sep, with otherwise unchanged settings, and scored on all of validation. Sequence model against LightGBM on these rows shows whether history helps when both see mature rows; the main run against this run shows what the time shift cost. This check is reported as an ablation and does not enter model selection or the combination test.
9. **Comparison entry:** the sequence model enters the comparison as the average of the three seeds' probabilities, labelled as such, with the single-seed mean and standard deviation shown beside it so the effect of averaging is visible. This was chosen before any results.

**Compute:** laptop with an RTX 3060 (6 GB) for light work. Heavier runs on Kaggle (Tesla T4, 16 GB), with a short interactive probe for speed and memory before every long background run.

**Expectation to state openly:** a single transaction carries little signal on this dataset. Most of it is in how accounts behave across transactions, and part of the measured performance rests on signals specific to the simulator. Results are therefore given as a ladder with the reason for each step, and compared against the results reported in the dataset paper.

**Results.** Measured results live in the notebooks: the baseline ladder and ablations in `02_baselines.ipynb`, and every model overall and by laundering group in `model_comparison.ipynb`. They are collected in `docs/MODELLING.md` at the end of step 4, and the README carries one headline table. Test scores were measured only after the model choice was locked.

**Results that drive design decisions:**
1. **Hub sent laundering is not detectable on current inputs.** It is about 13% of laundering in validation, and every model tried misses it, including the sequence model that reads each account's recent history: for a hub, the last 64 transactions cover only minutes. This caps recall at about 87%, sets the hub sender risk in section 14, and leaves an open question for the agent design in section 4.
2. **Licence and serving shape the choice of served model.** TabPFN-3.5 is under a non-commercial licence (fine-tuned or distilled versions count as derivatives) and needs a GPU to score, so it is a research benchmark only. The sequence transformer, chosen on validation, has its own weights and is small enough for a CPU, so it is the served model (decided 21 Nov 2026). Serving it needs each account's recent transactions at scoring time, which LightGBM's trailing counts also need in a simpler form; the demo bundles a sample of transactions with enough history for the accounts it shows. A score is explained by the past transactions that raise it most, found by removing each in turn and scoring again, since a transformer has no built-in feature importance.
3. **Simulator specific signals are strong** (payment format, the pair count, bank country). They are ablated, and the commercial path in section 16 depends on knowing which parts of the method would transfer.
4. **The test days contain a shift.** 9 Sep has about 36% more transactions than any validation day, from about 70% more distinct senders, and every model's precision collapses on it, while 10 Sep scores as on validation. Drift of this kind is what the drift check in step 9 must detect, so it monitors the number of active accounts and the score distribution, not only feature averages.

**Still to do in the modelling track:**
1. Test set predictions for TabPFN-3.5 and ModernBERT on Kaggle, and for Qwen after the weekly quota resets, added to the test table for reporting only.
2. `docs/MODELLING.md`.
3. A targeted experiment on hub sent laundering **[confirm whether still planned]**.

**Supporting data built for the agents:** a small synthetic watchlist of sanctioned or high risk banks and entities, sampled from the accounts file and documented as synthetic. The dataset has no sanctions information of its own.

**Stretch goals, only if on schedule after step 4 (section 12):**
- Run the same pipeline on IEEE-CIS, to show a result on real data.
- Use the LI-Small variant as a harder test set.
- A larger TabPFN context (for example 400,000 rows), under the same memory-only rule.
- TabPFN-2, the version licensed for commercial use: the same 202,530 training rows split into 27 bags of about 9,940 rows (within its 10,000 row limit), each scored on validation and averaged, at default settings.

The training pipeline is written to be dataset agnostic from the start, so these stay cheap to add.

**SageMaker use:** one training job for the final HuggingFace fine-tune, with the artifact registered in the model registry. Iteration happens locally or on free notebook GPUs (Kaggle).

## 6. Agent evaluation harness

Part of the 30% multi-agent effort. This is the main evidence for AI engineer roles.

**Test set:** 50 to 100 labelled cases. Laundering cases are drawn from the 370 attempts in the patterns file, spread across the 8 typologies. They are balanced with clean cases, watchlist hits, and ambiguous cases.

| What is measured | How |
|---|---|
| Routing correctness | Expected path against actual path through the graph |
| Flagging quality | Precision and recall against case labels |
| Tool use | Whether the right tools were called for the case type |
| Summary faithfulness | Deterministic checks that amounts, account IDs, and flags in the summary match case facts, plus an LLM judge for unsupported claims |
| Guardrails | Adversarial inputs, such as prompt injection inside a transaction memo field |
| Cost and latency | Tokens and seconds per case |

**In CI:** a small fixed subset runs on every push using recorded LLM responses, so the pipeline costs nothing. The full set runs on demand against Bedrock.

## 7. Non-goals

- Not a production financial crime system. Results are illustrative.
- Not regulatorily compliant. The AML and SAR logic is plausible, not audited.
- No real customer data. All data is synthetic.
- Not multi-tenant. One bank, one analyst role.
- Not high throughput. Demo scale only.
- No always-on paid infrastructure. The live demo uses free or scale to zero hosting (section 9).
- No feature store, streaming analytics, lakehouse, or batch ETL in Phase 1 (see section 13).
- No graph neural network in Phase 1 (see section 13).
- No fraud detection scorer in Phase 1 (see section 13, FRAML).
- No building of our own tabular foundation model. That is a possible separate project, not part of BankGuard.
- No mobile app or customer facing interface.

## 8. Skills matrix

| Skill | Feature that justifies it | Depth |
|---|---|---|
| Financial crime modelling | Baseline ladder, time based split, imbalance handling, graph derived features | Core |
| Model evaluation | PR-AUC, recall at fixed FPR, calibration, cost based threshold, error analysis, rules fixed in advance | Core |
| Experiment tracking (MLflow) | Every training run logged and compared | Core |
| HuggingFace fine-tuning | DistilBERT, ModernBERT, and Qwen3-0.6B with LoRA (PEFT) in the model comparison | Core |
| Tabular foundation models | TabPFN-3.5 as a benchmark, with its licence limits | Core |
| LangGraph orchestration | Conditional routing, loop, checkpoints, human interrupt | Core |
| LLM agents with tool use | Investigation agent chooses tools and decides when evidence is sufficient | Core |
| Agent evaluation | Labelled case set, faithfulness checks, regression suite in CI | Core |
| Semantic Kernel | Vector memory and shared tool plugins | Core |
| AWS Bedrock | LLM for the two agents | On demand |
| FastAPI | Analyst API with approve and reject endpoints | Core |
| React | Case review dashboard | Core (kept simple) |
| Testing and CI (Pytest, ruff, GitHub Actions) | Lint and unit tests on every push; model and agent regression tests to follow | Core |
| Continuous deployment (GitHub Actions) | Each push to main builds the container and updates the live demo | Core |
| Containers (Docker) | One image serves the model and the API for the live demo | Core |
| Kubernetes | Same manifests run on local kind and on EKS | Light locally, EKS on demand |
| SageMaker | One training job and model registry entry | On demand |
| Drift monitoring (Evidently) | Scheduled drift report on scores and inputs | Core (light) |
| Prometheus, Grafana, CloudWatch | Basic service and model metrics | Light |
| Cognito, IAM, SSM | Analyst login, least privilege roles, secrets | Light |
| Kinesis | Streaming ingestion variant | Documented |
| SageMaker Model Monitor | AWS native drift detection | Documented |

**Depth legend:**
- **Core:** built, tested, and shown in the demo. Runs locally at no cost.
- **On demand:** real AWS usage, switched on only when needed and paid from credits.
- **Light:** minimal working version, not a focus.
- **Documented:** described in the README with configuration, not running.

**Effort split:** about 30% modelling, 30% multi-agent system and evaluation, 40% application and infrastructure. The modelling share has run over this plan (section 10).

## 9. Architecture summary (v3)

| Layer | Phase 1 choice | Paid alternative, on demand or documented |
|---|---|---|
| Ingestion | Transaction replay script into SQS with a dead letter queue | Kinesis |
| Compute | Local Kubernetes (kind) | EKS |
| Agents and API | LangGraph and FastAPI in the cluster | Same images on EKS |
| Classifier serving | Model served in a pod | SageMaker Serverless Inference |
| LLM | Bedrock (Haiku class model), Ollama for development | None |
| Transaction store and vector memory | Postgres with pgvector in the cluster | Aurora, OpenSearch |
| Case store | DynamoDB | None needed |
| Auth and secrets | Cognito, SSM Parameter Store | Secrets Manager |
| Drift | Evidently | SageMaker Model Monitor |
| Observability | Prometheus, Grafana, CloudWatch within free limits | None needed |
| Registry and CI/CD | GitHub Actions, ECR Public, deploy to kind | Deploy to EKS |

Nothing runs on AWS when the project is idle.

**Public demo deployment.** Recruiters need a link that works without setup, so one deployment is public:

| Part | Choice |
|---|---|
| Front end | Static React build on free static hosting |
| API and model | One Docker container with FastAPI and the sequence transformer (three seeds, CPU-only torch), on a free or scale to zero host. The same image runs on local Kubernetes (kind) and, on demand, on EKS |
| Data | A small sample of transactions and accounts bundled with the service |
| Replay mode (default) | Visitors step through investigations recorded once. Scores and the dashboard are live, agent reasoning is played back. Cost per visit is zero |
| Live mode (limited) | Fresh agent runs against Bedrock, with a small daily cap so cost has a hard ceiling |
| Delivery | GitHub Actions runs tests, builds the image, and deploys on each push to main |

The host is chosen when the thin end to end version is built, after checking current free tiers. The first public version has no agents: it scores a chosen transaction and shows the main reasons.

## 10. Constraints

| Constraint | Value |
|---|---|
| AWS account | Personal, Paid plan since 12 Nov 2026 |
| Credits | $136.65 remaining, expiring 20 May 2027 |
| Extra credits available | Up to $60 from three unfinished activities, deadline about 20 Nov 2026 |
| Cost guardrail | Monthly budget of $10 with alerts at 50% and 100% actual and 100% forecasted, credits excluded from the calculation |
| Expected Phase 1 spend | A few dollars in total (Bedrock tokens, one SageMaker job, optional EKS demo days), all from credits |
| Idle cost | $0 |
| Live demo cost | $0 for visitors in replay mode. Live agent runs are capped per day |
| Free GPU | Kaggle, about 30 GPU hours a week, shared queue at busy times |
| Hard rule | Tear down any EKS cluster or endpoint the same day it is created |
| Hard date | All AWS resources deleted or the account closed before 20 May 2027 |
| Time budget | Originally 6 to 8 weeks at 10 to 15 hours a week. The modelling track has run longer than planned and the step 4 model list was widened on purpose, so the remaining steps are tracked by order, not by week **[confirm]** |
| Repo | Public, GitHub (`sap6011/bank-guard`), Sap as the only contributor |

There is no hard spending cap on a Paid plan, so the budget alerts and the teardown rule are the protection against charges.

## 11. Demo script (three minutes)

**Setup (0:00 to 0:20):** architecture diagram and one sentence on what the system does.

**Model evidence (0:20 to 1:00):** MLflow comparison of the model ladder, from the rules to TabPFN. PR-AUC and recall at fixed FPR for each model, the chosen threshold, and one line from the error analysis.

**Agents at work (1:00 to 2:00):** run the replay script. Show the graph trace: a low risk transaction ends early, a high risk one goes to investigation. Show the investigation agent choosing tools, then the drafted case summary.

**Human review (2:00 to 2:30):** the analyst opens the case in the dashboard and approves. The graph resumes and the case closes.

**Evaluation and close (2:30 to 3:00):** agent evaluation results table and a passing CI run. Point to the repo.

## 12. Build order and status

Revised on 16 Nov 2026, status updated 19 Nov 2026. A thin end to end version and its deployment pipeline come before the multi-agent system, so a live link exists early and every later change ships through the same pipeline.

| Step | Deliverable | Status |
|---|---|---|
| 1 | Baselines notebook: EDA, split, features, baseline ladder, ablations, error analysis | Done, closing summary included. Targeted hub sender experiment moved to step 4 **[confirm]** |
| 2 | Packaging: notebook code moved into `src/bankguard/`, with tests and MLflow tracking | Done. Data, features, evaluate, model, train and text modules, pytest suite, ruff, MLflow. Installable from GitHub, which the Kaggle notebooks use |
| 3 | Continuous integration: GitHub Actions runs the tests on every push | Done. Lint (ruff) then pytest on every push to main and on pull requests |
| 4 | Alternative models in their own notebooks, model selection, one final test set score, `docs/MODELLING.md` | In progress. DistilBERT, ModernBERT (2 and 4 epochs), TabPFN-3.5-Fast and the sequence model (main run) done. Qwen3-0.6B LoRA and the sequence model's time shift check done; TabPFN-2 moved to stretch goals. Combination test done (it scored below the sequence model), the sequence model chosen, and its test score measured, with LightGBM and DistilBERT for reference. Test scores for the Kaggle models and `docs/MODELLING.md` remain |
| 5 | Thin end to end version: saved model behind a FastAPI service, one basic page, a container | Not started |
| 6 | Continuous deployment: GitHub Actions builds and deploys the container. First live link | Not started |
| 7 | Multi-agent system: LangGraph graph, tools, two agents, loop, human interrupt | Not started |
| 8 | Agent evaluation harness and labelled case set, with a subset in CI, `docs/AGENT_EVAL.md` | Not started |
| 9 | Polish: replay mode, dashboard, drift check, README, demo video, v3 architecture diagram, AWS resources torn down | Not started |

Modelling came first on purpose. It is the part reviewers for these roles weigh most, and the agents need a real model to call.

## 13. Phase 2 roadmap (about six months after Phase 1)

Not in scope now. Listed so the Phase 1 boundaries read as deliberate.

| Capability | What it adds |
|---|---|
| Graph neural network scorer | Entity risk score from the transaction graph, alongside the tabular model, on the same IBM AML data. Phase 1 error analysis sets its targets: recall at a 0.1% false positive rate, hub sent laundering, and multi-hop typologies such as cycles, stacks and random chains |
| Parallel fraud scorer (FRAML) | A fraud classifier trained on IEEE-CIS runs alongside the AML scorer on every transaction. The supervisor combines both risk scores. Each scorer is evaluated on its own dataset, because no public dataset labels the same transactions for both fraud and laundering. No router classifier: fraud and laundering can co-occur, so both checks always run |
| Kafka (MSK) | Open source streaming path |
| Batch ETL on Spark | Raw data to curated feature tables |
| Lakehouse (Iceberg or Delta) | Analytical layer separate from the operational store |
| Feature store | Consistent features for training and serving |
| Streaming analytics | Real time aggregates such as account velocity |
| Pre-filter before the agents | A cheap model screens most events so agents see only suspicious ones |
| Batch orchestration | Airflow or Step Functions for ETL and retraining |
| Automated retraining | Drift triggers a retraining job and a staged rollout |

**Phase 1 choices that keep Phase 2 additive:**
1. Queue based ingestion, so new consumers can be added without changing upstream code.
2. The classifier sits behind one scoring interface that can hold several scorers, so models can be swapped or added (including the FRAML fraud scorer).
3. The graph is independent of the storage layer.
4. Tools are Semantic Kernel plugins, so new ones can be added without rewriting agents.
5. Kubernetes manifests work on both kind and EKS.

## 14. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Unexpected AWS charge | Low | High | $10 budget with alerts, same day teardown, hard date of 20 May 2027 |
| Simulator specific signals inflate scores (payment format, pair count, bank country, date) | High | Medium | Each is ablated and results are given as a ladder; split within 1 to 10 Sep; raw date never used as a feature |
| Public demo runs up LLM cost | Medium | High | Replay mode by default, a daily cap on live runs, and the $10 budget alert |
| Hub sent laundering stays undetected | High | Medium | Every model so far misses it, including the sequence model. One targeted experiment on new hub sender information; if it fails, the recall ceiling of about 87% is stated plainly |
| Scores are modest on a hard dataset | Medium | Low | Expected. Compare with the dataset paper and lead with error analysis by typology |
| Graph derived features leak future information | Medium | High | Compute them only from transactions earlier than the row being scored; add a test for it |
| Best model cannot be served (licence or GPU need, as with TabPFN-3.5) | Medium | Low | Report it as the research benchmark, serve a model with usable weights in the demo, and explain the choice in `docs/MODELLING.md` |
| Rules bent after seeing results | Medium | High | Evaluation rules fixed in advance (section 5) and kept even when a result lands just outside them |
| Kaggle GPU quota or queue delays runs | Medium | Medium | Probe before every long run, split long jobs into training and scoring runs, check the weekly quota before launching |
| Step 4 model list keeps growing | Medium | High | The list in section 5 is closed for Phase 1. New models go to stretch goals or Phase 2 |
| Agent evaluation set too small to be meaningful | Medium | Medium | At least 50 cases across four categories; report counts with every score |
| Scope creep | High | High | Non-goals in section 7; cuts go to Phase 2 |
| Local machine cannot run the full stack | Medium | Medium | Keep the cluster small; run Bedrock instead of a local model for demos; heavy training on Kaggle |
| Multi-agent design looks contrived | Medium | Medium | Only two LLM agents, each justified in section 4 |
| Infrastructure consumes the schedule | Medium | High | A thin end to end version is deployed first and grown in place, so infrastructure work is spread out and always has a working result |

## 15. Items to confirm

1. The remaining time budget, now that the modelling track has run longer than planned.
2. Whether the targeted hub sender experiment is still planned, and where it lives.
3. The host for the public demo, to be chosen at step 5 after checking current free tiers.
4. The daily cap on live agent runs in the public demo.
5. Whether SageMaker Serverless Inference should be switched on for the demo recording, to show SageMaker serving as well as training.
6. Whether the React dashboard should stay minimal, as assumed here.

## 16. Possible commercial path

Not in scope for Phase 1 or Phase 2. Recorded because it affects design choices now.

**Design choices kept open on purpose:**
- The packaged pipeline is dataset agnostic: loading, the split, features and evaluation take a schema, so a real institution's data could replace the synthetic set.
- Every alert is to carry the features behind its score and a record of what the agents did, since compliance users need explanations and an audit trail.
- Signals specific to the simulator are identified and ablated, so it is clear which parts of the method would be expected to transfer.
- Model licences are recorded with each model. TabPFN-3.5 is non-commercial; TabPFN-2 allows commercial use with attribution.

**Known gaps before any commercial use:**
- Results on synthetic data do not transfer directly to real banks. Real data requires partnerships and compliance work.
- The dataset licence terms must be checked for commercial use.
- TabPFN-3.5 would need a paid licence; the other models use weights that allow commercial use or are trained from scratch.
- The AML and SAR logic is plausible, not audited (section 7).
