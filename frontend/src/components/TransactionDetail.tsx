import { useEffect, useState } from 'react'
import { getTransaction, type PastTransfer, type TransactionDetail as Detail } from '../api'
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

interface Props {
  id: number
}

export default function TransactionDetail({ id }: Props) {
  const [detail, setDetail] = useState<Detail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let current = true
    getTransaction(id)
      .then((result) => {
        if (current) {
          setDetail(result)
          setError(null)
        }
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
  if (!detail || detail.id !== id) {
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
            <span key={a.budget} className={a.flagged ? 'tag alert' : 'tag quiet'}>
              {formatBudget(a.budget)} budget: {a.flagged ? 'alert' : 'no alert'}
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
      <ReasonTable rows={raising} empty="None: no past transfer raised the score." />
      <h3>Past transfers that lowered it</h3>
      <ReasonTable rows={lowering} empty="None." />
      <p className="muted">
        Effect: how much the score drops when that past transfer is removed and the transfer is scored again. A
        negative effect means the score would rise without it.
      </p>
    </section>
  )
}

function ReasonTable({ rows, empty }: { rows: PastTransfer[]; empty: string }) {
  if (rows.length === 0) return <p className="muted">{empty}</p>
  return (
    <table className="reasons">
      <thead>
        <tr>
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
          <tr key={`${p.side}-${p.slot}`}>
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