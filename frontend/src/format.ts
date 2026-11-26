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

export const GROUP_SHORT: Record<Group, string> = {
  patterned: 'Patterned',
  quiet: 'Unpatterned, quiet',
  hub: 'Unpatterned, hub',
  normal: 'Normal',
}

// ISO 4217 codes for the dataset's currency names; Bitcoin has no ISO code, BTC is the usual one.
const CURRENCY_CODES: Record<string, string> = {
  'US Dollar': 'USD',
  Euro: 'EUR',
  'UK Pound': 'GBP',
  Yuan: 'CNY',
  Yen: 'JPY',
  Rupee: 'INR',
  'Swiss Franc': 'CHF',
  'Canadian Dollar': 'CAD',
  'Australian Dollar': 'AUD',
  'Mexican Peso': 'MXN',
  'Brazil Real': 'BRL',
  Ruble: 'RUB',
  Shekel: 'ILS',
  'Saudi Riyal': 'SAR',
  Bitcoin: 'BTC',
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export function formatTime(time: string): string {
  // "2022-09-02 00:09" -> "2 Sep 00:09"; every transfer is in September 2022, so the year is left out.
  const [date, clock] = time.split(' ')
  const [, month, day] = date.split('-')
  return `${Number(day)} ${MONTHS[Number(month) - 1]} ${clock}`
}

export function formatAmount(amount: number, currency: string): string {
  const digits = amount < 1 ? 6 : 2 // small Bitcoin amounts need more decimals
  const code = CURRENCY_CODES[currency] ?? currency
  return `${amount.toLocaleString('en-US', { maximumFractionDigits: digits })} ${code}`
}

export function formatPercentile(percentile: number): string {
  return `${percentile.toFixed(2)}%`
}

export function formatEffect(effect: number): string {
  return `${effect >= 0 ? '+' : ''}${effect.toFixed(4)}`
}

export function formatBudget(budget: number): string {
  return `${(budget * 100).toLocaleString('en-US')}%`
}

export function formatRate(rate: number): string {
  const digits = rate > 0 && rate < 0.01 ? 2 : 1 // keep small rates such as 0.19% distinguishable
  return `${(rate * 100).toFixed(digits)}%`
}