import { useEffect, useState } from 'react'
import {
  getTransaction,
  whatIf,
  type Alert,
  type PastTransfer,
  type Side,
  type TransactionDetail as Detail,
  type WhatIfResponse,
} from '../api'
import {
  formatAmount,
  formatBudget,
  formatEffect,
  formatPercentile,
  GROUP_LABELS,
  SAMPLE_LABELS,
} from '../format'

const TOP_RAISING = 5
const TOP_LOWERING = 3

const keyOf = (p: { side: Side; slot: number }) => `${p.side}-${p.slot}`

interface Props {
  id: number
}

export default function TransactionDetail({ id }: Props) {
  const [detail, setDetail] = useState<Detail | null>(null)
  const [error, setError] = useState<string | null>(null)
  // What-if edits. App remounts this component for each transfer (key={id}), so they start empty.
  const [amountText, setAmountText] = useState('')
  const [removed, setRemoved] = useState<Set<string>>(new Set())
  const [result, setResult] = useState<WhatIfResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [whatIfError, setWhatIfError] = useState<string | null>(null)

  useEffect(() => {
    let current = true
    getTransaction(id)
      .then((d) => {
        if (current) setDetail(d)
      })
      .catch((e: Error) => {
        if (current) setError(e.message)
      })
    return () => {
      current = false
    }
  }, [id])

  if (error) {
    return (
      <section className="panel detail">
        <p className="error">{error}</p>
      </section>
    )
  }
  if (!detail) {
    return (
      <section className="panel detail">
        <p>Scoring transfer #{id} and each of its past transfers...</p>
      </section>
    )
  }

  const { transfer } = detail
  // past_transfers arrive largest effect first.
  const raising = detail.past_transfers.filter((p) => p.effect > 0).slice(0, TOP_RAISING)
  const lowering = detail.past_transfers.filter((p) => p.effect < 0).slice(-TOP_LOWERING).reverse()
  const counts = { sender: 0, receiver: 0 }
  for (const p of detail.past_transfers) counts[p.side] += 1

  function toggle(key: string) {
    setResult(null)
    setRemoved((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  async function runWhatIf() {
    if (!detail) return
    const amount = amountText.trim() === '' ? null : Number(amountText)
    if (amount !== null && !(amount > 0)) {
      setWhatIfError('Enter a positive amount, or leave the box empty to keep the original.')
      return
    }
    setRunning(true)
    setWhatIfError(null)
    try {
      const removedList = detail.past_transfers
        .filter((p) => removed.has(keyOf(p)))
        .map(({ side, slot }) => ({ side, slot }))
      setResult(await whatIf(id, { amount, removed: removedList }))
    } catch (e) {
      setWhatIfError((e as Error).message)
    } finally {
      setRunning(false)
    }
  }

  function reset() {
    setAmountText('')
    setRemoved(new Set())
    setResult(null)
    setWhatIfError(null)
  }

  return (
    <section className="panel detail">
      <h2>Transfer #{detail.id}</h2>
      <p className="mono">
        {transfer.from_account} → {transfer.to_account}
      </p>
      <p>
        {formatAmount(transfer.amount, transfer.currency)} on {transfer.time}{' '}
        <span className="muted">({transfer.payment_format}, not a model input)</span>
      </p>

      <div className="facts">
        <div>
          <span className="label">Rank</span>
          <strong>{formatPercentile(detail.percentile)}</strong>
          <span className="muted">raw score {detail.score.toFixed(4)}</span>
        </div>
        <div>
          <span className="label">Alerts</span>
          {detail.alerts.map((a) => (
            <span key={a.budget}>
              {formatBudget(a.budget)} budget: <AlertTag alert={a} />
            </span>
          ))}
        </div>
        <div>
          <span className="label">Actual</span>
          {detail.is_laundering ? (
            <span className="tag laundering">{GROUP_LABELS[detail.group]}</span>
          ) : (
            <span className="tag normal">Normal</span>
          )}
          {detail.typology && detail.typology !== 'No listed pattern' && (
            <span className="muted">{detail.typology}</span>
          )}
        </div>
      </div>
      <p className="muted">
        In the demo as: {SAMPLE_LABELS[detail.sample]}. History read by the model: {counts.sender} past transfers
        of the sender, {counts.receiver} of the receiver.
      </p>

      <h3>Past transfers that raised the score</h3>
      <ReasonTable rows={raising} empty="None: no past transfer raised the score." removed={removed} onToggle={toggle} />
      <h3>Past transfers that lowered it</h3>
      <ReasonTable rows={lowering} empty="None." removed={removed} onToggle={toggle} />
      <p className="muted">
        Effect: how much the score drops when that past transfer is removed and the transfer is scored again. A
        negative effect means the score would rise without it.
      </p>

      <h3>What if?</h3>
      <p className="muted">Change the amount, tick past transfers above to remove them, then score again.</p>
      <div className="whatif">
        <label>
          Amount in {transfer.currency}
          <input
            type="number"
            min="0"
            step="any"
            value={amountText}
            placeholder={String(transfer.amount)}
            onChange={(e) => {
              setAmountText(e.target.value)
              setResult(null)
            }}
          />
        </label>
        <button onClick={runWhatIf} disabled={running}>
          {running ? 'Scoring...' : 'Score again'}
        </button>
        <button onClick={reset} disabled={running}>
          Reset
        </button>
        <span className="muted">
          {removed.size} past transfer{removed.size === 1 ? '' : 's'} removed
        </span>
      </div>
      {whatIfError && <p className="error">{whatIfError}</p>}
      {result && <Comparison detail={detail} result={result} />}
    </section>
  )
}

function AlertTag({ alert }: { alert: Alert }) {
  return <span className={alert.flagged ? 'tag alert' : 'tag quiet'}>{alert.flagged ? 'alert' : 'no alert'}</span>
}

interface ReasonTableProps {
  rows: PastTransfer[]
  empty: string
  removed: Set<string>
  onToggle: (key: string) => void
}

function ReasonTable({ rows, empty, removed, onToggle }: ReasonTableProps) {
  if (rows.length === 0) return <p className="muted">{empty}</p>
  return (
    <table className="reasons">
      <thead>
        <tr>
          <th>Remove</th>
          <th>History</th>
          <th>Time</th>
          <th>From</th>
          <th>To</th>
          <th className="num">Amount</th>
          <th className="num">Effect</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((p) => (
          <tr key={keyOf(p)}>
            <td>
              <input
                type="checkbox"
                checked={removed.has(keyOf(p))}
                onChange={() => onToggle(keyOf(p))}
                aria-label="Remove this past transfer in the what-if"
              />
            </td>
            <td>{p.side === 'sender' ? "Sender's" : "Receiver's"}</td>
            <td>{p.transfer.time}</td>
            <td className="mono">{p.transfer.from_account}</td>
            <td className="mono">{p.transfer.to_account}</td>
            <td className="num">{formatAmount(p.transfer.amount, p.transfer.currency)}</td>
            <td className={`num ${p.effect > 0 ? 'up' : 'down'}`}>{formatEffect(p.effect)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Comparison({ detail, result }: { detail: Detail; result: WhatIfResponse }) {
  return (
    <table className="reasons comparison">
      <thead>
        <tr>
          <th></th>
          <th className="num">Original</th>
          <th className="num">What-if</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Rank</td>
          <td className="num">{formatPercentile(detail.percentile)}</td>
          <td className="num">{formatPercentile(result.percentile)}</td>
        </tr>
        <tr>
          <td>Raw score</td>
          <td className="num">{detail.score.toFixed(4)}</td>
          <td className="num">{result.score.toFixed(4)}</td>
        </tr>
        {detail.alerts.map((a, i) => (
          <tr key={a.budget}>
            <td>{formatBudget(a.budget)} budget</td>
            <td className="num">
              <AlertTag alert={a} />
            </td>
            <td className="num">
              <AlertTag alert={result.alerts[i]} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}