import { useEffect, useState } from 'react'
import { getModelCard, type ModelCard } from './api'

export default function App() {
  const [card, setCard] = useState<ModelCard | null>(null)
  const [error, setError] = useState<string | null>(null)

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
      </header>
      {error && <p className="error">Could not reach the API: {error}</p>}
      {!card && !error && <p>Loading the model; the first request can take a few seconds.</p>}
      {card && (
        <section>
          <h2>{card.model}</h2>
          <p>
            Test PR-AUC {card.test.pr_auc.toFixed(3)} on {card.test_transfers.toLocaleString()} transfers
            ({card.test_laundering} laundering).
          </p>
        </section>
      )}
    </main>
  )
}