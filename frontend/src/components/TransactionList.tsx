import { useEffect, useState } from 'react'
import { listTransactions, type Group, type Sample, type TransactionPage } from '../api'
import { formatAmount, formatPercentile, formatTime, GROUP_LABELS, GROUP_SHORT, SAMPLE_LABELS } from '../format'

const PAGE_SIZE = 25

type AlertFilter = 'any' | 'yes' | 'no'

interface Props {
  selectedId: number | null
  onSelect: (id: number) => void
}

export default function TransactionList({ selectedId, onSelect }: Props) {
  const [group, setGroup] = useState<Group | ''>('')
  const [sample, setSample] = useState<Sample | ''>('')
  const [alert, setAlert] = useState<AlertFilter>('any')
  const [sort, setSort] = useState<'score' | 'time'>('score')
  const [offset, setOffset] = useState(0)
  const [page, setPage] = useState<TransactionPage | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let current = true // answers to a request replaced by a newer one are ignored
    listTransactions({
      group: group || undefined,
      sample: sample || undefined,
      alert: alert === 'any' ? undefined : alert === 'yes',
      sort,
      limit: PAGE_SIZE,
      offset,
    })
      .then((result) => {
        if (current) {
          setPage(result)
          setError(null)
        }
      })
      .catch((e: Error) => {
        if (current) setError(e.message)
      })
    return () => {
      current = false
    }
  }, [group, sample, alert, sort, offset])

  const total = page?.total ?? 0

  return (
    <section className="panel">
      <div className="filters">
        <label>
          Group
          <select
            value={group}
            onChange={(e) => {
              setGroup(e.target.value as Group | '')
              setOffset(0)
            }}
          >
            <option value="">All</option>
            {Object.entries(GROUP_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Why in the demo
          <select
            value={sample}
            onChange={(e) => {
              setSample(e.target.value as Sample | '')
              setOffset(0)
            }}
          >
            <option value="">All</option>
            {Object.entries(SAMPLE_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Alert
          <select
            value={alert}
            onChange={(e) => {
              setAlert(e.target.value as AlertFilter)
              setOffset(0)
            }}
          >
            <option value="any">Any</option>
            <option value="yes">Alerted</option>
            <option value="no">Not alerted</option>
          </select>
        </label>
        <label>
          Sort
          <select
            value={sort}
            onChange={(e) => {
              setSort(e.target.value as 'score' | 'time')
              setOffset(0)
            }}
          >
            <option value="score">Highest risk first</option>
            <option value="time">By time</option>
          </select>
        </label>
      </div>

      <p className="muted">
        Rank: share of validation transfers scoring at or below this one. Alert: crosses the threshold set on
        validation to flag 1% of normal transfers.
      </p>

      {error && <p className="error">{error}</p>}
      {!page && !error && <p>Loading transfers...</p>}
      {page && (
        <>
          <p className="muted">
            {total.toLocaleString()} transfers
            {total > 0 && `, showing ${offset + 1} to ${Math.min(offset + PAGE_SIZE, total)}`}
          </p>
          <div className="table-wrap">
            <table className="transactions">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>From</th>
                  <th>To</th>
                  <th className="num">Amount</th>
                  <th>Actual</th>
                  <th className="num">Rank</th>
                  <th>Alert</th>
                </tr>
              </thead>
              <tbody>
                {page.items.map((item) => (
                  <tr
                    key={item.id}
                    className={item.id === selectedId ? 'selected' : undefined}
                    onClick={() => onSelect(item.id)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') onSelect(item.id)
                    }}
                    tabIndex={0}
                  >
                  <td>{formatTime(item.transfer.time)}</td>
                    <td className="mono">{item.transfer.from_account}</td>
                    <td className="mono">{item.transfer.to_account}</td>
                    <td className="num">{formatAmount(item.transfer.amount, item.transfer.currency)}</td>
                    <td>
                      {item.is_laundering ? (
                        <span className="tag laundering">{GROUP_SHORT[item.group]}</span>
                      ) : (
                        <span className="tag normal">Normal</span>
                      )}
                    </td>
                    <td className="num">{formatPercentile(item.percentile)}</td>
                    <td>{item.alert ? <span className="tag alert">Alert</span> : <span className="muted">No</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pager">
            <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>
              Previous
            </button>
            <button disabled={offset + PAGE_SIZE >= total} onClick={() => setOffset(offset + PAGE_SIZE)}>
              Next
            </button>
          </div>
        </>
      )}
    </section>
  )
}