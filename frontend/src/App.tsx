import { useEffect, useState } from 'react'
import { getModelCard, type ModelCard } from './api'
import TransactionList from './components/TransactionList'
import TransactionDetail from './components/TransactionDetail'
import ModelCardPanel from './components/ModelCardPanel'

export default function App() {
  const [card, setCard] = useState<ModelCard | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<number | null>(null)

  useEffect(() => {
    getModelCard()
      .then(setCard)
      .catch((e: Error) => setError(e.message))
  }, [])

  return (
    <main className="page">
      <header>
        <h1>BankGuard</h1>
        <p>Money laundering scores for synthetic bank transfers, with the past transfers behind each score.</p>
        {card && (
          <p className="muted">
            {card.model}. Test PR-AUC {card.test.pr_auc.toFixed(3)} on {card.test_transfers.toLocaleString()}{' '}
            transfers ({card.test_laundering} laundering).
          </p>
        )}
      </header>
      {card && <ModelCardPanel card={card} />}
      {error && <p className="error">Could not reach the API: {error}</p>}
      <div className="layout">
        <TransactionList selectedId={selectedId} onSelect={setSelectedId} />
        {selectedId === null ? (
          <section className="panel detail">
            <p className="muted">Pick a transfer to see its score and the past transfers behind it.</p>
          </section>
        ) : (
          <TransactionDetail key={selectedId} id={selectedId} />
        )}
      </div>
    </main>
  )
}