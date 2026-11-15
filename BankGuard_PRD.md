# BankGuard: Product Requirements Document

**Status:** Draft v4, 16 Nov 2026. Supersedes draft v3.1. Adds a public live demo, a revised build order (a thin end to end version before the multi-agent system, with CI/CD from that point), the modelling results measured so far, and a note on a possible commercial path. Draft v3.1 added a parallel fraud scorer (FRAML) to the Phase 2 roadmap. Draft v3 moved the dataset from PaySim to the IBM AML dataset (HI-Small).
**Matches:** Architecture v3 (zero idle cost), with the public demo deployment described in section 9.
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
3. **Score** laundering risk with the best model from the modelling track (deterministic model call).
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
- The modelling results shape the agent design. About 13% of laundering in validation is sent by hub accounts and cannot be ranked by the model (section 5). How such cases reach investigation, for example through rules or sampling, is an open design question for this stage.

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
4. **Self transfers are common.** 591,347 rows move money from an account to itself. EDA decides how to treat them.
5. **Two columns are both named `Account`.** They are renamed to sender and receiver accounts on load.

**Method:**
1. **Exploratory analysis** with a written summary of class balance, time structure, payment formats, currencies, and leakage risks.
2. **Time based split within 1 to 10 Sep.** Proposed: train on 1 to 6 Sep, validate on 7 to 8 Sep, test on 9 to 10 Sep (about 64%, 19%, and 17% of rows). The 11 to 18 Sep tail is excluded from headline metrics and reported separately. No random split. The raw date is never a feature.
3. **Features:**
   - Transaction level: log amount, amount relative to the training median for its currency, hour of day, and a self transfer flag.
   - Graph derived, computed only from earlier timestamps and excluding self transfers: incoming and outgoing counts for sender and receiver over trailing 1 and 3 day windows, and prior transfers between the same pair.
   - Counterparty breadth: recent transfers to and from new counterparties.
   - Account attributes from the accounts file: entity type, accounts per entity, same owner and same bank flags, and bank country.
4. **Baseline ladder:** two one line rules (ACH, new counterparty), logistic regression, then LightGBM, then a fine-tuned HuggingFace model. All baselines live in `02_baselines.ipynb`. The HuggingFace model has its own notebook.
5. **Imbalance handling and stability:** logistic regression uses balanced class weights. LightGBM is trained unweighted with L2 regularisation and early stopping on validation PR-AUC, after a fixed length run without them proved unstable. Headline models are retrained across seeds.
6. **Metrics:** PR-AUC as the primary metric, recall at a fixed false positive rate, calibration, and a cost based decision threshold. ROC-AUC is reported but not used for selection, because it is misleading at this class balance.
7. **Ablation:** signals that may depend on how the data was generated are reported separately: payment format, payment currency, the pair count, and bank country.
8. **Experiment tracking:** MLflow for every run, with parameters, metrics, and artifacts.
9. **Error analysis:** recall by laundering typology, and separately for patterned laundering, unpatterned laundering from quiet senders, and unpatterned laundering from hub senders.
10. **Model selection:** the best model on the validation set is the one served. If the HuggingFace model loses to gradient boosting, the README says so and explains why.

**Expectation to state openly:** a single transaction carries little signal on this dataset. Most of it is in how accounts behave across transactions, and part of the measured performance rests on signals specific to the simulator. Results are therefore given as a ladder with the reason for each step, and compared against the results reported in the dataset paper.

**Results to date (validation set, 16 Nov 2026).** The test set has not been used.

| Model | PR-AUC | Recall at a 1% false positive rate |
|---|---|---|
| Rule: flag every ACH transaction | 0.007 | Not usable at this budget |
| Rule: flag every new counterparty | 0.016 | Not usable at this budget |
| Logistic regression, transaction features | 0.002 | 2% |
| LightGBM, transaction features | 0.007 | 12% |
| LightGBM with graph counts | 0.270 | 66% |
| With counterparty breadth | 0.302 | 67% |
| With owner attributes | 0.341 | 76% |
| With bank country | 0.450 | 80% |

None of the trained models above use payment format or payment currency. For comparison, the graph count model with both added scores 0.446.

**What the results show:**
1. **Graph derived features carry most of the signal.** They raise PR-AUC about forty times over transaction features alone.
2. **Several strong signals are tied to the simulator.** Payment format (ACH holds 87% of laundering), the pair count (only 4.6% of transfers go to a counterparty not paid in the previous 3 days), and bank country. Each is ablated. Without the pair count the graph model scores 0.104.
3. **Results are stable across seeds.** For example 0.331 to 0.342 with owner attributes, and 0.449 to 0.451 with bank country.
4. **An early LightGBM setup was unstable** and its numbers were corrected in the repository history. The cause, the fix, and the corrected results are documented in the notebook.
5. **Error analysis by group, at a 1% false positive rate, for the model with owner attributes:** 91% recall on laundering in a listed pattern, 76% on unpatterned laundering from quiet senders, and about 2% on unpatterned laundering from hub senders.
6. **Hub sent laundering sets a ceiling.** It is 13% of laundering in validation, goes to known counterparties at normal amounts, and is not separable on current features. Recall cannot exceed about 87% unless it is detected.

**Still to do in the modelling track:** a targeted experiment on hub sent laundering, a closing summary, packaging the notebook code into `src/bankguard/` with tests, MLflow tracking, the HuggingFace model, and one final score on the test set.

**Supporting data built for the agents:** a small synthetic watchlist of sanctioned or high risk banks and entities, sampled from the accounts file and documented as synthetic. The dataset has no sanctions information of its own.

**Stretch goals, only if on schedule after M2:**
- Run the same pipeline on IEEE-CIS, to show a result on real data.
- Use the LI-Small variant as a harder test set.

The training pipeline is written to be dataset agnostic from the start, so both stay cheap to add.

**SageMaker use:** one training job for the final HuggingFace fine-tune, with the artifact registered in the model registry. Iteration happens locally or on free notebook GPUs.

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
- No mobile app or customer facing interface.

## 8. Skills matrix

| Skill | Feature that justifies it | Depth |
|---|---|---|
| Financial crime modelling | Baseline ladder, time based split, imbalance handling, graph derived features | Core |
| Model evaluation | PR-AUC, recall at fixed FPR, calibration, cost based threshold, error analysis | Core |
| Experiment tracking (MLflow) | Every training run logged and compared | Core |
| HuggingFace fine-tuning | Fine-tuned candidate in the baseline ladder | Core |
| LangGraph orchestration | Conditional routing, loop, checkpoints, human interrupt | Core |
| LLM agents with tool use | Investigation agent chooses tools and decides when evidence is sufficient | Core |
| Agent evaluation | Labelled case set, faithfulness checks, regression suite in CI | Core |
| Semantic Kernel | Vector memory and shared tool plugins | Core |
| AWS Bedrock | LLM for the two agents | On demand |
| FastAPI | Analyst API with approve and reject endpoints | Core |
| React | Case review dashboard | Core (kept simple) |
| Testing and CI (Pytest, GitHub Actions) | Unit tests, model regression tests, agent regression suite, run on every push | Core |
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

**Effort split:** about 30% modelling, 30% multi-agent system and evaluation, 40% application and infrastructure.

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
| API and model | One container with FastAPI and the saved LightGBM model, on a free or scale to zero host |
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
| Hard rule | Tear down any EKS cluster or endpoint the same day it is created |
| Hard date | All AWS resources deleted or the account closed before 20 May 2027 |
| Time budget | Originally 6 to 8 weeks at 10 to 15 hours a week. The modelling track ran longer than planned, so the remaining steps are tracked by order, not by week **[confirm]** |
| Repo | Public, GitHub |

There is no hard spending cap on a Paid plan, so the budget alerts and the teardown rule are the protection against charges.

## 11. Demo script (three minutes)

**Setup (0:00 to 0:20):** architecture diagram and one sentence on what the system does.

**Model evidence (0:20 to 1:00):** MLflow comparison of the baseline ladder. PR-AUC and recall at fixed FPR for each model, the chosen threshold, and one line from the error analysis.

**Agents at work (1:00 to 2:00):** run the replay script. Show the graph trace: a low risk transaction ends early, a high risk one goes to investigation. Show the investigation agent choosing tools, then the drafted case summary.

**Human review (2:00 to 2:30):** the analyst opens the case in the dashboard and approves. The graph resumes and the case closes.

**Evaluation and close (2:30 to 3:00):** agent evaluation results table and a passing CI run. Point to the repo.

## 12. Build order and status

Revised on 16 Nov 2026. A thin end to end version and its deployment pipeline now come before the multi-agent system, so a live link exists early and every later change ships through the same pipeline.

| Step | Deliverable | Status |
|---|---|---|
| 1 | Baselines notebook: EDA, split, features, baseline ladder, ablations, error analysis | Nearly done. Hub sender experiment and closing summary remain |
| 2 | Packaging: notebook code moved into `src/bankguard/`, with tests and MLflow tracking | Not started |
| 3 | Continuous integration: GitHub Actions runs the tests on every push | Not started |
| 4 | HuggingFace model in its own notebook, model selection, one final test set score, `docs/MODELLING.md` | Not started |
| 5 | Thin end to end version: saved model behind a FastAPI service, one basic page, a container | Not started |
| 6 | Continuous deployment: GitHub Actions builds and deploys the container. First live link | Not started |
| 7 | Multi-agent system: LangGraph graph, tools, two agents, loop, human interrupt | Not started |
| 8 | Agent evaluation harness and labelled case set, with a subset in CI, `docs/AGENT_EVAL.md` | Not started |
| 9 | Polish: replay mode, dashboard, drift check, README, demo video, AWS resources torn down | Not started |

Modelling came first on purpose. It is the part reviewers for these roles weigh most, and the agents need a real model to call. Steps 2 to 4 are kept short so the visible product is reached soon.

## 13. Phase 2 roadmap (about six months after Phase 1)

Not in scope now. Listed so the Phase 1 boundaries read as deliberate.

| Capability | What it adds |
|---|---|
| Graph neural network scorer | Entity risk score from the transaction graph, alongside the tabular model, on the same IBM AML data. Phase 1 error analysis sets its targets: recall at a 0.1% false positive rate, and multi-hop typologies such as cycles, stacks and random chains |
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
| Hub sent laundering stays undetected | High | Medium | One targeted experiment. If it fails, the recall ceiling of about 87% is stated plainly |
| Scores are modest on a hard dataset | Medium | Low | Expected. Compare with the dataset paper and lead with error analysis by typology |
| Graph derived features leak future information | Medium | High | Compute them only from transactions earlier than the row being scored; add a test for it |
| HuggingFace model loses to gradient boosting | Medium | Low | Expected and acceptable. Serve the winner and explain the result |
| Agent evaluation set too small to be meaningful | Medium | Medium | At least 50 cases across four categories; report counts with every score |
| Scope creep | High | High | Non-goals in section 7; cuts go to Phase 2 |
| Local machine cannot run the full stack | Medium | Medium | Keep the cluster small; run Bedrock instead of a local model for demos |
| Multi-agent design looks contrived | Medium | Medium | Only two LLM agents, each justified in section 4 |
| Infrastructure consumes the schedule | Medium | High | A thin end to end version is deployed first and grown in place, so infrastructure work is spread out and always has a working result |

## 15. Items to confirm

1. The remaining time budget, now that the modelling track has run longer than planned.
2. The host for the public demo, to be chosen at step 5 after checking current free tiers.
3. The daily cap on live agent runs in the public demo.
4. Whether SageMaker Serverless Inference should be switched on for the demo recording, to show SageMaker serving as well as training.
5. Whether the React dashboard should stay minimal, as assumed here.

## 16. Possible commercial path

Not in scope for Phase 1 or Phase 2. Recorded because it affects design choices now.

**Design choices kept open on purpose:**
- The pipeline is dataset agnostic: loading, the split, features and evaluation take a schema, so a real institution's data could replace the synthetic set.
- Every alert carries the features behind its score and a record of what the agents did, since compliance users need explanations and an audit trail.
- Signals specific to the simulator are identified and ablated, so it is clear which parts of the method would be expected to transfer.

**Known gaps before any commercial use:**
- Results on synthetic data do not transfer directly to real banks. Real data requires partnerships and compliance work.
- The dataset licence terms must be checked for commercial use.
- The AML and SAR logic is plausible, not audited (section 7).
