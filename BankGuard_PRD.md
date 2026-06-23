# BankGuard — Product Requirements Document

> **Status:** Draft v1. Filled in with default assumptions for a portfolio project. Items marked 🟡 are best-guesses Saptarshi should confirm or override.

---

## 1. One-paragraph brief

BankGuard is a portfolio project that demonstrates production-grade GenAI engineering — specifically multi-agent orchestration with LangGraph, AWS-native MLOps, and full-stack delivery — through a plausible bank fraud detection system. It is reviewed primarily by senior engineering hiring managers and ML platform leads at fintech and GenAI-focused companies. Success means a reviewer can, in under five minutes, see that Saptarshi can design, build, and ship an enterprise-shaped GenAI system end-to-end. The project lives as a public GitHub repo with a recorded 3-minute demo video and a one-click `terraform apply` to deploy live; it stays *deployed* only during active job-hunt periods, with an idle cost ceiling of **under $10/month** when torn down to documentation-only mode.

---

## 2. Audience and reviewer journey

**Primary reviewer:** Senior engineering hiring manager or staff/principal ML platform engineer screening candidates for GenAI, ML platform, or AI engineering roles at mid-to-large fintech or AI-product companies.

What they care about:
- Production thinking — does this person understand observability, CI/CD, security, cost?
- Architectural judgment — did they pick the right tools, or just the most fashionable ones?
- Breadth credibly demonstrated — not just name-dropped on a CV
- A working demo, not slideware

What turns them off:
- Tutorial-grade code that reads like a copy of someone's blog post
- Buzzword soup with no defensible reason for each choice
- A broken or unclear demo
- An architecture diagram that doesn't match the code

**Secondary reviewer:** AWS solutions architects or technical interviewers who'll deep-dive on specific services during a loop.

**Reviewer journey:**
1. They land on the GitHub README. They see a one-paragraph pitch, the architecture diagram inline, a "watch the 3-minute demo" link at the top, and a clear skills-demonstrated table.
2. They click the demo video. They see transactions streaming in, agents working, an analyst approving a flagged case, and a Grafana-style dashboard with model drift metrics.
3. If interested, they read `docs/ARCHITECTURE.md` (decisions and tradeoffs), `docs/PRD.md` (this document — itself a signal of product thinking), and skim the LangGraph supervisor code.
4. The thing that makes them want to talk to Saptarshi: the project demonstrates *judgment* — explicit non-goals, defensible tradeoffs, and a cost-aware deployment story. It looks like something built by an engineer, not a bootcamp grad.

---

## 3. Functional scope

The system does, end-to-end:
1. **Ingest** a stream of synthetic transaction events (event-driven, not REST submissions)
2. **Normalise and enrich** each transaction (ingestion agent)
3. **Score** fraud risk using a fine-tuned HuggingFace classifier served on SageMaker Serverless Inference (anomaly scoring agent)
4. **Flag** regulatory concerns — sanctions screening, AML rules, SAR triggers (regulatory flagging agent)
5. **Summarise** flagged cases into draft analyst reports using a Bedrock LLM with Guardrails (case summary agent)
6. **Pause** for human review — the LangGraph graph checkpoints; the case appears in the React dashboard
7. **Resume** when the analyst approves or rejects; decision persists to the case store
8. **Monitor** for model drift on a schedule; alerts surface in the observability dashboard

**Successful demo run (60-second happy path):**
A transaction stream is replayed at 5x speed. Within seconds, three cases are flagged. The analyst dashboard lights up with one high-priority case — a $47K wire to a sanctioned jurisdiction. The analyst clicks in, reads the AI-drafted summary, approves the SAR filing. The graph resumes, the decision is logged, the case moves to "closed" status. A side panel shows the classifier's feature attribution and the current drift score.

---

## 4. Non-goals

Explicit list of what BankGuard is **not**:

- **Not a real fraud detection system** — model accuracy is illustrative, no benchmark claims against industry baselines
- **Not multi-tenant** — single bank, single analyst persona, single role
- **Not regulatorily compliant** — SAR logic is plausible but not audited; no real bank should ship this
- **Not handling real PII** — all transaction data is synthetic (PaySim-derived)
- **Not high-throughput** — demo-scale only, single-shard streaming, sub-100 transactions per minute
- **Not production-hardened** — no DR, no multi-region, no SLA, no on-call rotation
- **Not building** a mobile app
- **Not building** a customer-facing UI — analyst dashboard only
- **Not building** a feature store *in v1* — features computed inline; feature store deferred to v2 roadmap (§11)
- **Not building** real-time model retraining *in v1* — scheduled batch retraining only; streaming retraining deferred to v2 roadmap (§11)
- **Not implementing** OAuth flows beyond Cognito's defaults
- **Not optimising** Bedrock token usage beyond reasonable prompt hygiene

---

## 5. Skills demonstration matrix

| Skill | Feature that justifies it | Depth |
|---|---|---|
| LangGraph multi-agent orchestration | Supervisor routes four specialist agents; HITL interrupts pause for analyst approval; checkpointing persists graph state | **Live** |
| AWS Bedrock + Guardrails | LLM calls for case summarisation; Guardrails filter the regulatory-adjacent output | **Live** (Claude Haiku to keep token cost ~cents per demo) |
| HuggingFace fine-tuning | Custom transaction classifier fine-tuned on PaySim synthetic fraud dataset | **Live** (one-time training, artifact in S3) |
| SageMaker | Training jobs (one-shot), model registry, Serverless Inference endpoint | **Live** (Serverless = pay-per-call, no idle cost) |
| Semantic Kernel | Memory abstraction + shared skill plugins (sanctions-lookup, geo-risk-lookup) callable from any agent | **Live** |
| FastAPI | Analyst-facing API — case queries, approval/reject endpoints, auth middleware | **Live** |
| React | Case review dashboard with approve/reject flow and drift-metric panel | **Live** |
| EKS | Production deployment target | **Documented + spin-up-on-demand** — Terraform deploys it; live only during demo recording; torn down between job-hunt windows |
| Prometheus + CloudWatch | Prometheus runs on EKS (live when EKS is up); CloudWatch always-on for metrics, logs, drift alarms | **Live (CloudWatch) / on-demand (Prometheus)** |
| Cognito + IAM | Analyst auth via Cognito; least-privilege IAM roles per agent and per service | **Live** |
| Kinesis | Streaming ingestion of transaction events | **Documented + on-demand** — single shard during demos, swap to local replay for daily dev |
| GitHub Actions → ECR → EKS | CI pipeline runs pytest + regression suite, builds image, pushes to ECR, deploys to EKS | **Live** (pipeline always runs on push; deploy step gated by manual trigger to control cost) |
| DynamoDB | Case store with review lifecycle | **Live** (free tier covers it forever) |
| Aurora Serverless v2 (Postgres + pgvector) | Transaction store and vector memory store consolidated | **Live** (scales to zero ACU when idle) 🟡 alternative: keep RDS + OpenSearch separate if reviewer-target weights "more services on the diagram" higher |
| SageMaker Model Monitor | Scheduled drift detection on classifier inputs and outputs | **Live** (daily schedule, cheap) |

**Depth legend:**
- **Live** = running in AWS, reviewer can hit it
- **Documented + on-demand** = Terraform + README explain it; spun up only during demo recording, torn down after
- **Stubbed** = mocked, with notes on what the real version would do

---

## 6. Constraints

| Constraint | Value |
|---|---|
| Cost ceiling (idle, between job hunts) | **< $10/month** (achieved by tearing down EKS, Kinesis, SageMaker endpoint; leaving only DynamoDB, S3, ECR, CloudWatch logs) |
| Cost ceiling (active, during job hunt) | **< $80/month** (EKS control plane + 1 small node + Kinesis 1 shard + Aurora minimum ACU + CloudWatch) |
| Cost during demo recording | One-off ~$5–10 for a few hours of full-stack uptime |
| Time budget to v1 | **6–8 weeks** at evenings + weekends 🟡 confirm |
| Estimated effort | ~10–15 hours/week 🟡 confirm |
| Hard deadline | None imposed; soft target = before next active job hunt 🟡 |
| AWS account type | Personal account 🟡 confirm |
| Repo | **Public** GitHub |
| Billing alarms | $5, $20, $50 — set before first `terraform apply` |

---

## 7. Open questions blocking architecture (proposed answers)

| Question | Proposed answer | Rationale |
|---|---|---|
| Hosting model | **Hybrid** — repo + recorded demo always available; live deploy spun up during active job hunts | Best of both: low idle cost, real live URL when it matters |
| EKS live or documented-only | **Spin-up-on-demand** | Saves $73/month idle without dropping EKS from skill list |
| ML model real or stubbed | **Real** — fine-tuned on PaySim | Stubbed classifiers are a portfolio downgrade |
| Dataset | **PaySim** (Kaggle, synthetic, well-known) | Public baselines exist; no PII risk; small enough to fine-tune cheaply |
| Inference | **SageMaker Serverless Inference** | Pay-per-call, scales to zero; cold start (~5–10s) acceptable for analyst-review workflow (not customer-facing) |
| Vector store | **pgvector on Aurora Serverless v2** | Consolidates with transaction store; scales to zero; defensible architectural simplification 🟡 reversible if you want OpenSearch on the diagram for skill-coverage reasons |
| Streaming | **Kinesis live during demos, local replay script for dev** | Keeps Kinesis on the skill list without paying for an always-on shard |
| Agents at runtime | **Containers on EKS** | LangGraph supervisor benefits from in-memory graph state; matches the "enterprise deployment" narrative better than Lambda |

---

## 8. Demo script (3 minutes)

**Setup (0:00–0:20)** — Screen shows architecture diagram. Voiceover: "BankGuard is a multi-agent fraud detection system I built to demonstrate production GenAI on AWS. Four specialist agents, a LangGraph supervisor, a fine-tuned HuggingFace classifier on SageMaker, and a human-in-the-loop analyst dashboard. Here's how it runs."

**Act 1 — ingestion (0:20–0:50)** — Switch to terminal. Run the transaction replay script. Show events landing in Kinesis (CloudWatch metrics ticking up). Cut to the LangGraph trace view showing the ingestion agent normalising events.

**Act 2 — scoring and flagging (0:50–1:40)** — Show the supervisor routing to the anomaly scoring agent, which calls the SageMaker endpoint. Surface one transaction that scores high. Show the regulatory flagging agent matching it against a sanctions list. Show the case summary agent drafting a SAR-ready summary via Bedrock — point out Guardrails filtering the output.

**Act 3 — human review (1:40–2:30)** — Switch to React dashboard. Analyst sees the flagged case with the AI-drafted summary, the classifier's confidence, and feature attribution. Analyst clicks Approve. LangGraph resumes, decision persists to DynamoDB.

**Close (2:30–3:00)** — Cut to the Grafana dashboard showing request latency, drift score, and a CI/CD pipeline run from the most recent commit. Voiceover: "Full repo is linked below. Skills demonstrated: LangGraph orchestration, Bedrock, SageMaker fine-tuning and serving, EKS, Semantic Kernel, full CI/CD, observability, and AWS security. Happy to walk through any of it."

---

## 9. Milestones

| Milestone | Deliverable | Target |
|---|---|---|
| M1 — PRD locked | This doc reviewed and frozen | Week 0 |
| M2 — Architecture v3 | Diagram aligned to PRD (pgvector, Serverless Inference, on-demand EKS) | Week 1 |
| M3 — Local end-to-end skeleton | LangGraph graph wired with stub agents; FastAPI + React talking; runs on laptop | Week 2 |
| M4 — Real ML model | PaySim fine-tuning job runs on SageMaker; classifier registered; Serverless endpoint live | Week 4 |
| M5 — Cloud deployment | Terraform deploys full stack; CI/CD green; live URL works | Week 6 |
| M6 — Demo recorded | 3-minute video published; README polished; PRD and ARCHITECTURE docs in repo | Week 7 |
| M7 — Idle teardown | Terraform `destroy` for paid services; repo + video remain | Week 8 |

---

## 10. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cost overrun while building | Med | High | Billing alarms at $5/$20/$50 set before first deploy; teardown checklist; daily cost dashboard for the first two weeks |
| Scope creep | High | High | Non-goals list in §4; weekly check against milestones; cuts go to "v2 backlog" not v1 |
| Reviewer can't follow demo | Med | High | Demo script written before build (this doc, §8); test video on a non-technical friend before publishing |
| ML model performs poorly | Med | Med | PaySim has public baselines; target AUC ≥ 0.90 (achievable per Kaggle solutions); fallback = present it honestly with the metric |
| LangGraph + Semantic Kernel split looks contrived | Med | Med | ARCHITECTURE.md has a one-paragraph crisp justification; SK is constrained to memory + 2 specific skills, nothing else |
| EKS cold spin-up wastes demo-recording time | Low | Med | Practice teardown/spin-up cycle once before recording; have a fallback `docker compose` local stack |
| Cognito + IAM setup eats more time than expected | Med | Low | Time-box to 1 week; if blocked, ship with a hard-coded analyst user and document the production auth path |
| HuggingFace fine-tuning doesn't converge in budget | Low | Med | Use a small distilled model (DistilBERT-style) or tabular MLP; PaySim is small enough that training is cheap |
| Job hunt timing changes | Med | Low | The hybrid hosting model handles this — spin up when needed, tear down when not |

---

## 11. v2 roadmap — deferred ~6 months post-v1

> **Status: not in scope for v1.** This section exists to (a) signal to reviewers that BankGuard's v1 boundaries are deliberate, not naive, and (b) preserve forward-compatible decisions in v1 so v2 is additive, not a rewrite.

**v2 scope themes:** big-data processing, real ETL, streaming analytics, lakehouse, and a proper feature store. Driven by the realisation that v1's "agents process every event" pattern doesn't scale past demo volume — at production scale, you need cheap pre-filtering and a real analytical data layer.

| v2 capability | What it adds | Why deferred from v1 |
|---|---|---|
| **Kafka (MSK) alongside Kinesis** | Open-source-native streaming path; reviewers see both AWS-native and OSS-native ingestion | Kinesis covers the streaming skill demonstration for v1; adding Kafka in v1 is double-counting the same skill |
| **Batch ETL on Spark (EMR Serverless or Glue)** | Real ETL job: PaySim raw → S3 → Spark → curated feature tables → SageMaker training | v1 trains directly on PaySim CSV; sufficient for demo, but not a credible big-data story |
| **Iceberg/Delta lakehouse on S3** | Analytical layer separate from operational Aurora; CDC from Aurora feeds the lake | v1 collapses operational + analytical into Aurora to fit cost ceiling |
| **Feature store (Feast or SageMaker Feature Store)** | Anomaly scoring agent pulls features from the store instead of computing inline; enables training/serving consistency | v1 non-goal — adds infra without changing the agent narrative |
| **Streaming analytics (Flink on Kinesis Data Analytics)** | Real-time aggregates (per-account velocity, geo-anomaly windows) that feed the scoring agent as additional features | Parallel consumer of the existing event log; clean v2 add |
| **Pre-filter before agent pipeline** | Cheap streaming classifier (e.g. XGBoost on velocity features) screens out 99% of events; agents only see the suspicious 1% | Demonstrates *production* judgment about when LLM-agents are too expensive — but only meaningful at real volume |
| **Airflow or Step Functions for batch orchestration** | Schedules ETL DAGs, retraining, lakehouse maintenance | v1 has no batch jobs to orchestrate; only meaningful once ETL exists |
| **Real-time / online retraining loop** | Drift detection triggers an automated retraining job; new model registered, A/B rolled out | v1 has Model Monitor surfacing drift, but retraining is manual |
| **Iceberg time-travel for case audit** | Regulatory-grade audit trail — "show me the exact state of the transaction store when this SAR was filed" | v1's DynamoDB case log is sufficient for the demo; lakehouse time-travel is overkill at v1 scale |

### Forward-compatible decisions already in v1

These v1 calls are made specifically so v2 is additive rather than a rewrite:

1. **Event-driven ingress (Kinesis)** rather than REST submission of transactions — means v2 can fan out the same event log to Spark, Flink, lakehouse, and the pre-filter without touching upstream code
2. **SageMaker model registry** for the v1 classifier — v2 retraining loops can register new versions against the same registry
3. **LangGraph state graph** decoupled from data layer specifics — when Aurora is split into operational-only + lakehouse in v2, the agents don't change
4. **Semantic Kernel skills as plugins** — v2 can add a "feature-store-lookup" skill without rewriting agents
5. **Terraform-first deployment** — v2 additions land as new modules, not as manual AWS console clicks layered on v1

### What v2 would *not* do

- Not multi-tenant even in v2 — the project stays single-bank
- Not real PII — still synthetic data, just more of it
- Not a real fraud product — accuracy still illustrative
- Not real-time customer-facing decisions — analyst-mediated workflow remains the model

### Estimated v2 timing

~6 months after v1 ships. Trigger conditions: (a) v1 has been used in actual interviews and feedback warrants expansion, or (b) Saptarshi wants to demonstrate big-data / data-engineering skills explicitly for a different role family (data platform, ML platform engineer at scale, etc.).

---



1. **Target reviewer persona** — I assumed fintech/AI-product hiring managers. If you're targeting a specific company or role family, this should be more specific.
2. **Time budget** — I assumed 6–8 weeks at 10–15 hrs/week. Adjust based on your actual availability.
3. **AWS account** — assumed personal. If employer-provided or sandbox, cost constraints might relax.
4. **Aurora + pgvector vs RDS + OpenSearch** — I chose consolidation for cost. If you'd rather have OpenSearch visible on the diagram for skill-coverage reasons, that's a one-line reversal.
5. **Cost ceilings** — I picked $10 idle / $80 active. If you have a different number in mind, every "live vs documented" call shifts.
