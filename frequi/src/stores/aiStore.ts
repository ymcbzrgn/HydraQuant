import { defineStore } from 'pinia';
import axios from 'axios';

export interface AIStatus {
    autonomy_level: number;
    active_model: string;
    daily_cost: number;
    cache_hit_rate: number;
    status?: string;
    uptime?: string;
}

export interface AISentiment {
    pair: string;
    sentiment_1h: number;
    sentiment_4h: number;
    sentiment_24h: number;
    fear_greed: number;
    source_count: number;
    last_update: string;
}

export interface AISignal {
    pair: string;
    signal: string;
    confidence: number;
    reasoning: string;
    timestamp: string;
    outcome?: string;
}

export interface AICostSummary {
    today_cost: number;
    models: Record<string, any>;
    budget_remaining: number;
}

export interface AIAutonomy {
    current_level: number;
    kelly_fraction: number;
    criteria: any;
    history: any[];
}

export interface AIRisk {
    portfolio_value: number;
    daily_budget: number;
    consumed: number;
    utilization_pct: number;
    active_positions: number;
}

export interface AIForgonePnl {
    total_forgone: number;
    weekly_summary: any;
    recent_signals: number;
}

export interface AIHealth {
    status: string;
    checks: Record<string, any>;
    alerts: string[];
}

export interface AIMetrics {
    rag_latency_avg_ms: number;
    llm_cost_today: number;
    cache_hit_rate: number;
    total_decisions: number;
    error_rate: number;
    retrieval_count: number;
    active_pairs: string[];
    uptime_hours: number;
    last_updated: string;
}

export interface AIPortfolio {
    stake_currency: string;
    total_balance: number;
    free_balance: number;
    in_trades: number;
    assets: Record<string, any>;
    total_portfolio_usd: number;
    updated_at: string;
}

// Detect AI API URL: same host as the browser, port 8890
function getDefaultAiApiUrl(): string {
    if (typeof window !== 'undefined' && window.location) {
        const host = window.location.hostname;
        return `http://${host}:8890`;
    }
    return 'http://localhost:8890';
}

export const useAiStore = defineStore('ai', {
    state: () => ({
        status: null as AIStatus | null,
        sentiment: {} as Record<string, AISentiment>,
        signals: [] as AISignal[],
        costSummary: null as AICostSummary | null,
        autonomy: null as AIAutonomy | null,
        risk: null as AIRisk | null,
        forgonePnl: null as AIForgonePnl | null,
        confidenceHistory: [] as any[],
        health: null as AIHealth | null,
        metrics: null as AIMetrics | null,
        portfolio: null as AIPortfolio | null,
        loading: false,
        error: null as string | null,
        aiApiUrl: getDefaultAiApiUrl(),
        lastFetchTime: null as Date | null,
        isAiOnline: false,
    }),

    getters: {
        aiStatusBadge: (state) => {
            if (!state.isAiOnline) return 'offline';
            return state.health?.status || 'unknown';
        },
        portfolioValueFormatted: (state) => {
            const val = state.portfolio?.total_portfolio_usd || state.risk?.portfolio_value || 0;
            return `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        },
    },

    actions: {
        async fetchStatus() {
            try {
                const { data } = await axios.get<AIStatus>(`${this.aiApiUrl}/api/ai/status`);
                this.status = data;
                this.isAiOnline = true;
            } catch (err: any) {
                this.isAiOnline = false;
                console.error('Failed to fetch AI Status', err);
            }
        },
        async fetchSentiment(pair: string) {
            try {
                const encodedPair = encodeURIComponent(pair);
                const { data } = await axios.get<AISentiment>(`${this.aiApiUrl}/api/ai/sentiment/${encodedPair}`);
                if (data && data.pair) {
                    this.sentiment[data.pair] = data;
                }
            } catch (err: any) {
                console.error(`Failed to fetch AI Sentiment for ${pair}`, err);
            }
        },
        async fetchSignals(limit: number = 20) {
            try {
                const { data } = await axios.get<AISignal[]>(`${this.aiApiUrl}/api/ai/signals?limit=${limit}`);
                this.signals = data;
            } catch (err: any) {
                console.error('Failed to fetch AI Signals', err);
            }
        },
        async fetchCostSummary() {
            try {
                const { data } = await axios.get<AICostSummary>(`${this.aiApiUrl}/api/ai/cost`);
                this.costSummary = data;
            } catch (err: any) {
                console.error('Failed to fetch AI Cost', err);
            }
        },
        async fetchAutonomy() {
            try {
                const { data } = await axios.get<AIAutonomy>(`${this.aiApiUrl}/api/ai/autonomy`);
                this.autonomy = data;
            } catch (err: any) {
                console.error('Failed to fetch AI Autonomy', err);
            }
        },
        async fetchRisk() {
            try {
                const { data } = await axios.get<AIRisk>(`${this.aiApiUrl}/api/ai/risk`);
                this.risk = data;
            } catch (err: any) {
                console.error('Failed to fetch AI Risk', err);
            }
        },
        async fetchForgonePnl() {
            try {
                const { data } = await axios.get<AIForgonePnl>(`${this.aiApiUrl}/api/ai/forgone`);
                this.forgonePnl = data;
            } catch (err: any) {
                console.error('Failed to fetch AI Forgone PNL', err);
            }
        },
        async fetchConfidenceHistory(pair?: string, days: number = 7) {
            try {
                let url = `${this.aiApiUrl}/api/ai/confidence-history?days=${days}`;
                if (pair) {
                    url += `&pair=${encodeURIComponent(pair)}`;
                }
                const { data } = await axios.get<any[]>(url);
                this.confidenceHistory = data;
            } catch (err: any) {
                console.error('Failed to fetch Confidence History', err);
            }
        },
        async fetchHealth() {
            try {
                const { data } = await axios.get<AIHealth>(`${this.aiApiUrl}/api/ai/health`);
                this.health = data;
                this.isAiOnline = true;
            } catch (err: any) {
                this.isAiOnline = false;
                console.error('Failed to fetch AI Health', err);
            }
        },
        async fetchMetrics() {
            try {
                const { data } = await axios.get<AIMetrics>(`${this.aiApiUrl}/api/ai/metrics`);
                this.metrics = data;
            } catch (err: any) {
                console.error('Failed to fetch AI Metrics', err);
            }
        },
        async fetchPortfolio() {
            try {
                const { data } = await axios.get<AIPortfolio>(`${this.aiApiUrl}/api/ai/portfolio`);
                this.portfolio = data;
            } catch (err: any) {
                console.error('Failed to fetch AI Portfolio', err);
            }
        },
        async fetchAll() {
            this.loading = true;
            this.error = null;
            try {
                await Promise.all([
                    this.fetchStatus(),
                    this.fetchHealth(),
                    this.fetchMetrics(),
                    this.fetchSignals(50),
                    this.fetchCostSummary(),
                    this.fetchAutonomy(),
                    this.fetchRisk(),
                    this.fetchForgonePnl(),
                    this.fetchPortfolio(),
                ]);
                this.lastFetchTime = new Date();
            } catch (err: any) {
                this.error = 'Error fetching AI data';
            } finally {
                this.loading = false;
            }
        },
    },
});
