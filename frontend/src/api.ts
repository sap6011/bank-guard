// Types mirror the Pydantic models in src/bankguard/api.py.

export type Group = 'patterned' | 'quiet' | 'hub' | 'normal'
export type Sample = 'laundering' | 'top_normal' | 'random_normal'
export type Side = 'sender' | 'receiver'

export interface Transfer {
  time: string
  from_account: string
  to_account: string
  amount: number
  currency: string
  payment_format: string
}

export interface Alert {
  budget: number
  threshold: number
  flagged: boolean
}

export interface TransactionSummary {
  id: number
  transfer: Transfer
  is_laundering: boolean
  group: Group
  sample: Sample
  score: number
  percentile: number
  alert: boolean
}

export interface TransactionPage {
  total: number
  items: TransactionSummary[]
}

export interface PastTransfer {
  side: Side
  slot: number
  effect: number
  transfer: Transfer
}

export interface TransactionDetail {
  id: number
  transfer: Transfer
  is_laundering: boolean
  typology: string | null
  group: Group
  sample: Sample
  score: number
  percentile: number
  alerts: Alert[]
  past_transfers: PastTransfer[]
}

export interface Removal {
  side: Side
  slot: number
}

export interface WhatIfRequest {
  amount: number | null
  removed: Removal[]
}

export interface WhatIfResponse {
  amount: number | null
  removed: Removal[]
  original_score: number
  score: number
  percentile: number
  alerts: Alert[]
}

export interface Recall {
  fpr: number
  recall: number
}

export interface Metrics {
  pr_auc: number
  recall: Recall[]
}

export interface GroupRecall {
  group: Group
  laundering: number
  recall: Recall[]
}

export interface AlertCheck {
  budget: number
  threshold: number
  test_recall: number
  test_false_alarm_rate: number
}

export interface ModelCard {
  model: string
  data: string
  training: string
  validation: Metrics
  test: Metrics
  test_transfers: number
  test_laundering: number
  test_recall_by_group: GroupRecall[]
  alert_thresholds: AlertCheck[]
  limitations: string[]
}

export interface ListParams {
  group?: Group
  sample?: Sample
  alert?: boolean
  sort?: 'score' | 'time'
  limit?: number
  offset?: number
}

// In development this is /api (the Vite proxy); a deployed build sets VITE_API_URL.
const BASE = import.meta.env.VITE_API_URL ?? '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init)
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${await response.text()}`)
  }
  return (await response.json()) as T
}

export function listTransactions(params: ListParams = {}): Promise<TransactionPage> {
  const query = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) query.set(key, String(value))
  }
  return request(`/transactions?${query}`)
}

export function getTransaction(id: number): Promise<TransactionDetail> {
  return request(`/transactions/${id}`)
}

export function whatIf(id: number, body: WhatIfRequest): Promise<WhatIfResponse> {
  return request(`/transactions/${id}/what-if`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function getModelCard(): Promise<ModelCard> {
  return request('/model-card')
}