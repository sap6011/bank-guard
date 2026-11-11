# BankGuard: Product Requirements Document

**Status:** Draft v2, 12 Nov 2026. Supersedes draft v1.
**Matches:** Architecture v3 (zero idle cost).
**Naming:** "Phase 1" is the first complete build described here. "Phase 2" is the upgrade planned about six months later (section 13). Diagram versions (v1, v2, v3) are a separate numbering.

Items marked **[confirm]** are assumptions Sap should confirm or override.

## 1. Brief

BankGuard is a portfolio project that shows applied machine learning and AI engineering through a bank fraud detection system. It combines a rigorously evaluated fraud classifier with a LangGraph multi-agent investigation workflow that has human review built in. It targets data scientist, ML engineer, and AI engineer roles. Success means a reviewer can see, in under five minutes, evidence of sound modelling judgment, measurable agent quality, and the ability to ship the result. The project lives in a public GitHub repo with a recorded three minute demo. It runs locally at zero cost, with AWS services used on demand and paid from account credits.

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
1. README: one paragraph pitch, architecture diagram, demo link, and a results table (model comparison and agent evaluation scores).
2. Demo video: a flagged transaction moves through investigation to analyst approval.
3. Deeper read: `docs/MODELLING.md`, `docs/AGENT_EVAL.md`, `docs/ARCHITECTURE.md`, and this PRD.
4. The hook: every claim in the README is backed by a number and a notebook or test that reproduces it.

## 3. Functional scope

End to end, the system does the following:

1. **Ingest** synthetic transaction events from a replay script through a queue.
2. **Normalise and enrich** each event (deterministic code).
3. **Score** fraud risk with the best model from the modelling track (deterministic model call).
4. **Route** on the score: low risk ends the run, high risk goes to investigation.
5. **Investigate** with an LLM agent that chooses its own tools: sanctions lookup, account history query, and similar past cases from vector memory.
6. **Loop** when evidence is insufficient: the supervisor sends the case back for more evidence, up to a fixed limit.
7. **Summarise** the case into a draft analyst report with guardrails applied.
8. **Pause** for human review. The graph checkpoints and the case appears in the dashboard.
9. **Resume** when the analyst approves or rejects. The decision is stored.
10. **Monitor** score drift on a schedule.

**Happy path for the demo:** a replayed stream produces three flagged cases. One is a large transfer to a sanctioned counterparty. The investigation agent calls two tools, the summary agent drafts the report, the analyst approves, and the case closes.

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

**Capabilities demonstrated:** conditional routing, tool use, a bounded loop, vector memory, checkpointing, and a human interrupt with resume.

## 5. Modelling track

About 30% of the effort. This is the main evidence for data scientist and ML engineer roles.

**Dataset:** PaySim (synthetic mobile money transactions). Expected properties, to be confirmed on load: about 6.3 million rows, about 0.1% fraud, fraud limited to transfer and cash out transaction types, and an hourly time step.

**Method:**
1. **Exploratory analysis** with a written summary of class balance, transaction types, and leakage risks.
2. **Time based split** on the step column for train, validation, and test. No random split, to avoid leaking future behaviour.
3. **Baseline ladder:** logistic regression, then gradient boosting (LightGBM or XGBoost), then a fine-tuned HuggingFace model.
4. **Imbalance handling:** class weights first, resampling only if it helps on validation.
5. **Metrics:** PR-AUC as the primary metric, recall at a fixed false positive rate, calibration, and a cost based decision threshold. ROC-AUC is reported but not used for selection, because it is misleading at this class balance.
6. **Experiment tracking:** MLflow for every run, with parameters, metrics, and artifacts.
7. **Error analysis:** a write-up of which fraud the best model misses and why.
8. **Model selection:** the best model on the validation set is the one served. If the HuggingFace model loses to gradient boosting, the README says so and explains why.

**Known issue to handle openly:** PaySim's balance columns make fraud unusually easy to separate, so near perfect scores are common. The write-up will report results with and without those features, so the numbers are not read as better than they are.

**SageMaker use:** one training job for the final HuggingFace fine-tune, with the artifact registered in the model registry. Iteration happens locally or on free notebook GPUs.

## 6. Agent evaluation harness

Part of the 30% multi-agent effort. This is the main evidence for AI engineer roles.

**Test set:** 50 to 100 hand labelled synthetic cases covering clear fraud, clear non-fraud, sanctions hits, and ambiguous cases.

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

- Not a production fraud system. Results are illustrative.
- Not regulatorily compliant. The AML and SAR logic is plausible, not audited.
- No real customer data. All data is synthetic.
- Not multi-tenant. One bank, one analyst role.
- Not high throughput. Demo scale only.
- No always-on cloud deployment or live URL.
- No feature store, streaming analytics, lakehouse, or batch ETL in Phase 1 (see section 13).
- No graph neural network in Phase 1 (see section 13).
- No mobile app or customer facing interface.

## 8. Skills matrix

| Skill | Feature that justifies it | Depth |
|---|---|---|
| Fraud modelling | Baseline ladder, time based split, imbalance handling | Core |
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
| Testing and CI (Pytest, GitHub Actions) | Unit tests, model regression tests, agent regression suite | Core |
| Containers and Kubernetes | Same manifests run on local kind and on EKS | Core locally, EKS on demand |
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
| Ingestion | PaySim replay script into SQS with a dead letter queue | Kinesis |
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

## 10. Constraints

| Constraint | Value |
|---|---|
| AWS account | Personal, Paid plan since 12 Nov 2026 |
| Credits | $136.65 remaining, expiring 20 May 2027 |
| Extra credits available | Up to $60 from three unfinished activities, deadline about 20 Nov 2026 |
| Cost guardrail | Monthly budget of $10 with alerts at 50% and 100% actual and 100% forecasted, credits excluded from the calculation |
| Expected Phase 1 spend | A few dollars in total (Bedrock tokens, one SageMaker job, optional EKS demo days), all from credits |
| Idle cost | $0 |
| Hard rule | Tear down any EKS cluster or endpoint the same day it is created |
| Hard date | All AWS resources deleted or the account closed before 20 May 2027 |
| Time budget | 6 to 8 weeks at 10 to 15 hours a week **[confirm]** |
| Repo | Public, GitHub |

There is no hard spending cap on a Paid plan, so the budget alerts and the teardown rule are the protection against charges.

## 11. Demo script (three minutes)

**Setup (0:00 to 0:20):** architecture diagram and one sentence on what the system does.

**Model evidence (0:20 to 1:00):** MLflow comparison of the baseline ladder. PR-AUC and recall at fixed FPR for each model, the chosen threshold, and one line from the error analysis.

**Agents at work (1:00 to 2:00):** run the replay script. Show the graph trace: a low risk transaction ends early, a high risk one goes to investigation. Show the investigation agent choosing tools, then the drafted case summary.

**Human review (2:00 to 2:30):** the analyst opens the case in the dashboard and approves. The graph resumes and the case closes.

**Evaluation and close (2:30 to 3:00):** agent evaluation results table and a passing CI run. Point to the repo.

## 12. Milestones

| Milestone | Deliverable | Target |
|---|---|---|
| M0 | PRD and architecture v3 agreed; AWS guardrails set; credit activities done | Week 0 |
| M1 | EDA, time based split, baseline ladder, MLflow tracking | Weeks 1 to 2 |
| M2 | HuggingFace fine-tune, model selection, error analysis, `docs/MODELLING.md` | Week 3 |
| M3 | LangGraph graph: deterministic nodes, two agents, tools, loop, human interrupt | Week 4 |
| M4 | Agent evaluation harness and labelled case set, `docs/AGENT_EVAL.md` | Week 5 |
| M5 | FastAPI, React dashboard, local Kubernetes deployment | Week 6 |
| M6 | CI pipeline, drift check, basic observability, one SageMaker training job | Week 7 |
| M7 | Demo recorded, README finished, AWS resources torn down | Week 8 |

Modelling comes first on purpose. It is the part reviewers for these roles weigh most, and the agents need a real model to call.

## 13. Phase 2 roadmap (about six months after Phase 1)

Not in scope now. Listed so the Phase 1 boundaries read as deliberate.

| Capability | What it adds |
|---|---|
| Graph neural network scorer | Entity risk score from the transaction graph, alongside the tabular model, on a dataset with real graph structure (IEEE-CIS or Elliptic) |
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
2. The classifier sits behind one scoring interface, so models can be swapped or added.
3. The graph is independent of the storage layer.
4. Tools are Semantic Kernel plugins, so new ones can be added without rewriting agents.
5. Kubernetes manifests work on both kind and EKS.

## 14. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Unexpected AWS charge | Low | High | $10 budget with alerts, same day teardown, hard date of 20 May 2027 |
| PaySim results look too good to be credible | High | Medium | Report with and without the balance features; time based split; say so plainly |
| HuggingFace model loses to gradient boosting | Medium | Low | Expected and acceptable. Serve the winner and explain the result |
| Agent evaluation set too small to be meaningful | Medium | Medium | At least 50 cases across four categories; report counts with every score |
| Scope creep | High | High | Non-goals in section 7; cuts go to Phase 2 |
| Local machine cannot run the full stack | Medium | Medium | Keep the cluster small; run Bedrock instead of a local model for demos |
| Multi-agent design looks contrived | Medium | Medium | Only two LLM agents, each justified in section 4 |
| Infrastructure consumes the schedule | Medium | High | Infrastructure work starts in week 6, after modelling and agents are done |

## 15. Items to confirm

1. Time budget of 6 to 8 weeks at 10 to 15 hours a week.
2. "Phase 1" and "Phase 2" as the names for the project versions.
3. Whether SageMaker Serverless Inference should be switched on for the demo recording, to show SageMaker serving as well as training.
4. Whether the React dashboard should stay minimal, as assumed here.
