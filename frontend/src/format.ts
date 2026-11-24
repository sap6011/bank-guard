import type { Group, Sample } from './api'

export const GROUP_LABELS: Record<Group, string> = {
  patterned: 'Patterned laundering',
  quiet: 'Unpatterned laundering, quiet sender',
  hub: 'Unpatterned laundering, hub sender',
  normal: 'Normal',
}

export const SAMPLE_LABELS: Record<Sample, string> = {
  laundering: 'Laundering',
  top_normal: 'Top-scored normal (false alarms)',
  random_normal: 'Random normal',
}

export function formatAmount(amount: number, currency: string): string {
  const digits = amount < 1 ? 6 : 2 // small Bitcoin amounts need more decimals
  return `${amount.toLocaleString('en-US', { maximumFractionDigits: digits })} ${currency}`
}

export function formatPercentile(percentile: number): string {
  return `${percentile.toFixed(2)}%`
}