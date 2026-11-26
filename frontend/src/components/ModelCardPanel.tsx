import type { Metrics, ModelCard, Recall } from '../api'
import { formatBudget, formatRate, GROUP_LABELS } from '../format'

const FPRS = [0.001, 0.01]

function recallAt(recall: Recall[], fpr: number): string {
  const found = recall.find((r) => r.fpr === fpr)
  return found ? formatRate(found.recall) : 'n/a'
}

export default function ModelCardPanel({ card }: { card: ModelCard }) {
  const results: [string, Metrics][] = [
    ['Validation, 7 to 8 Sep', card.validation],
    [`Test, 9 to 10 Sep (${card.test_transfers.toLocaleString()} transfers)`, card.test],
  ]

  return (
    <details className="panel model-card">
      <summary>About the model: how well it does and where it fails</summary>
      <p>
        {card.model}. {card.data}. {card.training}.
      </p>

      <div className="card-grid">
        <div>
          <h3>Results</h3>
          <div className="table-wrap">
            <table className="reasons">
              <thead>
                <tr>
                  <th></th>
                  <th className="num">PR-AUC</th>
                  {FPRS.map((f) => (
                    <th key={f} className="num">
                      Recall at {formatBudget(f)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.map(([label, m]) => (
                  <tr key={label}>
                    <td>{label}</td>
                    <td className="num">{m.pr_auc.toFixed(3)}</td>
                    {FPRS.map((f) => (
                      <td key={f} className="num">
                        {recallAt(m.recall, f)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <h3>Test recall by group</h3>
          <div className="table-wrap">
            <table className="reasons">
              <thead>
                <tr>
                  <th>Group</th>
                  <th className="num">Laundering</th>
                  {FPRS.map((f) => (
                    <th key={f} className="num">
                      At {formatBudget(f)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {card.test_recall_by_group.map((g) => (
                  <tr key={g.group}>
                    <td>{GROUP_LABELS[g.group]}</td>
                    <td className="num">{g.laundering}</td>
                    {FPRS.map((f) => (
                      <td key={f} className="num">
                        {recallAt(g.recall, f)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <h3>Alert thresholds on the test days</h3>
          <div className="table-wrap">
            <table className="reasons">
              <thead>
                <tr>
                  <th>Set to flag</th>
                  <th className="num">Actually flagged</th>
                  <th className="num">Laundering caught</th>
                </tr>
              </thead>
              <tbody>
                {card.alert_thresholds.map((a) => (
                  <tr key={a.budget}>
                    <td>{formatBudget(a.budget)} of normal</td>
                    <td className="num">{formatRate(a.test_false_alarm_rate)} of normal</td>
                    <td className="num">{formatRate(a.test_recall)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <p className="muted">
        Recall at 1%: the share of laundering caught when 1% of normal transfers are flagged. Thresholds are set
        on validation, as a bank would set them from past data.
      </p>

      <h3>Limitations</h3>
      <ul className="limitations">
        {card.limitations.map((text) => (
          <li key={text}>{text}</li>
        ))}
      </ul>
    </details>
  )
}