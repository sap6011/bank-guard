"""The BankGuard scoring service.

Serves the sequence transformer over the demo bundle: a list of demo
transfers, one transfer's score with the past transfers behind it, and
what-if scoring. Paths come from BANKGUARD_MODEL_DIR and BANKGUARD_BUNDLE,
defaulting to the repo's models folder.
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from bankguard.serving import DemoScorer

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("BANKGUARD_MODEL_DIR", ROOT / "models" / "sequence"))
BUNDLE = Path(os.environ.get("BANKGUARD_BUNDLE", ROOT / "models" / "demo" / "bundle.npz"))

# Short codes for the laundering groups and sample reasons stored in the bundle.
GROUPS = {
    "patterned": "patterned",
    "unpatterned, quiet sender": "quiet",
    "unpatterned, hub sender": "hub",
    "normal": "normal",
}
SAMPLES = {"laundering": "laundering", "top-scored normal": "top_normal", "random normal": "random_normal"}
Group = Literal["patterned", "quiet", "hub", "normal"]
Sample = Literal["laundering", "top_normal", "random_normal"]
Side = Literal["sender", "receiver"]


class Transfer(BaseModel):
    time: str
    from_account: str
    to_account: str
    amount: float
    currency: str
    payment_format: str = Field(description="Shown for reference only; not a model input")


class Alert(BaseModel):
    budget: float = Field(description="Share of normal transfers flagged at this threshold, set on validation")
    threshold: float
    flagged: bool


class TransactionSummary(BaseModel):
    id: int
    transfer: Transfer
    is_laundering: bool
    group: Group
    sample: Sample = Field(description="Why this transfer is in the demo set")
    score: float = Field(description="Raw model score: ranks transfers, not a probability")
    percentile: float = Field(description="Percentage of validation transfers scoring at or below")
    alert: bool = Field(description="Flagged at the 1% alert budget")


class TransactionPage(BaseModel):
    total: int
    items: list[TransactionSummary]


class PastTransfer(BaseModel):
    side: Side
    slot: int
    effect: float = Field(description="Score with this past transfer minus score without it")
    transfer: Transfer


class TransactionDetail(BaseModel):
    id: int
    transfer: Transfer
    is_laundering: bool
    typology: str | None
    group: Group
    sample: Sample
    score: float
    percentile: float
    alerts: list[Alert]
    past_transfers: list[PastTransfer] = Field(description="Largest effect first")


class Removal(BaseModel):
    side: Side
    slot: int = Field(ge=0)


class WhatIfRequest(BaseModel):
    amount: float | None = Field(None, gt=0, description="New amount in the transfer's own currency")
    removed: list[Removal] = []


class WhatIfResponse(BaseModel):
    amount: float | None = Field(description="The amount used; null means unchanged")
    removed: list[Removal] = Field(description="The past transfers removed")
    original_score: float
    score: float
    percentile: float
    alerts: list[Alert]


class Recall(BaseModel):
    fpr: float = Field(description="Share of normal transfers flagged")
    recall: float = Field(description="Share of laundering caught at that rate")


class Metrics(BaseModel):
    pr_auc: float
    recall: list[Recall]


class GroupRecall(BaseModel):
    group: Group
    laundering: int
    recall: list[Recall]


class AlertCheck(BaseModel):
    budget: float = Field(description="Share of normal transfers the threshold was set to flag, on validation")
    threshold: float
    test_recall: float
    test_false_alarm_rate: float = Field(description="Share of test normal transfers it actually flagged")


class ModelCard(BaseModel):
    model: str
    data: str
    training: str
    validation: Metrics
    test: Metrics
    test_transfers: int
    test_laundering: int
    test_recall_by_group: list[GroupRecall]
    alert_thresholds: list[AlertCheck]
    limitations: list[str]


def build_model_card(scorer: DemoScorer) -> ModelCard:
    """The model card: computed results from the bundle plus the limitations they imply."""
    card = json.loads(str(scorer.data["model_card"]))
    for g in card["test_recall_by_group"]:
        g["group"] = GROUPS[g["group"]]
    hub = next(g for g in card["test_recall_by_group"] if g["group"] == "hub")
    hub_recall = max(r["recall"] for r in hub["recall"])
    loose = max(card["alert_thresholds"], key=lambda a: a["budget"])
    seeds = ", ".join(str(s) for s in scorer.prep["seeds"])
    length = scorer.prep["history_length"]
    return ModelCard(
        model=f"Sequence transformer, average of {len(scorer.prep['seeds'])} seeds ({seeds})",
        data="IBM Transactions for Anti Money Laundering, HI-Small (synthetic)",
        training="Trained on 1 to 6 Sep 2022, chosen on validation (7 to 8 Sep), tested once on 9 to 10 Sep",
        **card,
        limitations=[
            ("Scores rank transfers but are not probabilities: training kept every laundering transfer "
            "and a sample of normal ones, so laundering was about 1 in 80 training rows against about "
            "1 in 900 in reality. Read the percentile and the alerts instead."),
            (f"Laundering sent by hub accounts (at least 1,000 payments in the previous 3 days) is not "
            f"detected: at most {hub_recall:.0%} test recall. For a hub, its last {length} transfers "
            f"cover only minutes."),
            (f"Alert thresholds are set on validation. On the test days the {loose['budget']:.0%} budget "
            f"flagged {loose['test_false_alarm_rate']:.1%} of normal transfers, because 9 Sep brought "
            f"many more occasionally active accounts. A deployment needs drift monitoring."),
            ("Payment format, currency and bank country are left out on purpose: in this simulator "
            "they act as shortcuts that would not hold at a real bank."),
            (f"Explanations remove one past transfer at a time; a full {length}-event history does not "
            f"pull in an older transfer to replace it."),
            "The data is synthetic, so results do not transfer directly to real banks.",
        ],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models and the demo bundle once, before the first request."""
    scorer = DemoScorer(MODEL_DIR, BUNDLE)
    d = scorer.data
    scores = d["evaluated_score"]  # equal to live scores, checked in tests/test_serving.py
    app.state.scorer = scorer
    app.state.table = {
        "group": np.array([GROUPS[g] for g in d["group"]]),
        "sample": np.array([SAMPLES[s] for s in d["sample"]]),
        "score": scores,
        "percentile": scorer.percentile(scores),
        "alert": scorer.alerts(scores)[:, 0],
    }
    app.state.card = build_model_card(scorer)
    yield


app = FastAPI(
    title="BankGuard",
    description="Scores a bank transfer for money laundering and shows the past transfers behind the score.",
    version="0.1.0",
    lifespan=lifespan,
)


def find(request: Request, tx_id: int) -> DemoScorer:
    """The scorer, after checking that the demo transfer exists."""
    scorer = request.app.state.scorer
    if not 0 <= tx_id < len(scorer):
        raise HTTPException(404, f"No demo transaction {tx_id}")
    return scorer


def alert_list(scorer: DemoScorer, score: float) -> list[Alert]:
    d = scorer.data
    return [
        Alert(budget=float(b), threshold=float(t), flagged=bool(score >= t))
        for b, t in zip(d["alert_fpr"], d["alert_threshold"])
    ]


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check used by the host and by Kubernetes."""
    return {"status": "ok"}

@app.get("/model-card")
def model_card(request: Request) -> ModelCard:
    """What the served model is, how well it did, and where it fails."""
    return request.app.state.card

@app.get("/transactions")
def list_transactions(
    request: Request,
    group: Group | None = None,
    sample: Sample | None = None,
    alert: bool | None = None,
    sort: Literal["score", "time"] = "score",
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> TransactionPage:
    """Demo transfers, filtered and sorted, one page at a time."""
    scorer, t = request.app.state.scorer, request.app.state.table
    keep = np.ones(len(scorer), dtype=bool)
    if group:
        keep &= t["group"] == group
    if sample:
        keep &= t["sample"] == sample
    if alert is not None:
        keep &= t["alert"] == alert
    ids = np.flatnonzero(keep)  # already in time order
    if sort == "score":
        ids = ids[np.argsort(-t["score"][ids], kind="stable")]
    page = ids[offset:offset + limit]
    transfers = scorer.transfers(scorer.data["tx"][page])
    items = [
        TransactionSummary(
            id=int(i),
            transfer=transfer,
            is_laundering=bool(scorer.data["is_laundering"][i]),
            group=t["group"][i],
            sample=t["sample"][i],
            score=float(t["score"][i]),
            percentile=float(t["percentile"][i]),
            alert=bool(t["alert"][i]),
        )
        for i, transfer in zip(page, transfers)
    ]
    return TransactionPage(total=len(ids), items=items)


@app.get("/transactions/{tx_id}")
def get_transaction(tx_id: int, request: Request) -> TransactionDetail:
    """One demo transfer: live score, rank, alerts, and every past transfer's effect."""
    scorer = find(request, tx_id)
    d = scorer.data
    score, table = scorer.explain(tx_id)
    past = scorer.transfers(table["detail"].to_numpy()) if len(table) else []
    laundering = bool(d["is_laundering"][tx_id])
    return TransactionDetail(
        id=tx_id,
        transfer=scorer.transfers([d["tx"][tx_id]])[0],
        is_laundering=laundering,
        typology=str(d["typology"][tx_id]) if laundering else None,
        group=GROUPS[str(d["group"][tx_id])],
        sample=SAMPLES[str(d["sample"][tx_id])],
        score=float(score),
        percentile=float(scorer.percentile([score])[0]),
        alerts=alert_list(scorer, score),
        past_transfers=[
            PastTransfer(side=row.side, slot=int(row.slot), effect=float(row.effect), transfer=transfer)
            for row, transfer in zip(table.itertuples(), past)
        ],
    )


WHAT_IF_EXAMPLES = {
    "unchanged": {
        "summary": "No changes (returns the original score)",
        "value": {"amount": None, "removed": []},
    },
    "new amount": {
        "summary": "A different amount, in the transfer's own currency",
        "value": {"amount": 50000, "removed": []},
    },
    "remove a past transfer": {
        "summary": "Remove one past transfer (use a side and slot from GET /transactions/{id})",
        "value": {"amount": None, "removed": [{"side": "receiver", "slot": 35}]},
    },
}


@app.post("/transactions/{tx_id}/what-if")
def what_if(
    tx_id: int,
    request: Request,
    body: Annotated[WhatIfRequest, Body(openapi_examples=WHAT_IF_EXAMPLES)],
) -> WhatIfResponse:
    """Score a demo transfer again with a new amount and/or past transfers removed."""
    scorer = find(request, tx_id)
    d = scorer.data
    for r in body.removed:
        mask = d[f"{r.side}_mask"][tx_id]
        if r.slot >= len(mask) or not mask[r.slot]:
            raise HTTPException(422, f"{r.side} slot {r.slot} holds no past transfer")
    score = scorer.what_if(tx_id, amount=body.amount, removed=[(r.side, r.slot) for r in body.removed])
    return WhatIfResponse(
        amount=body.amount,
        removed=body.removed,
        original_score=float(d["evaluated_score"][tx_id]),
        score=score,
        percentile=float(scorer.percentile([score])[0]),
        alerts=alert_list(scorer, score),
    )