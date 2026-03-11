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
        loading: false,
        error: null as string | null,
        aiApiUrl: 'http://localhost:8890',
    }),

    actions: {
        async fetchStatus() {
            try {
                const { data } = await axios.get<AIStatus>(`${this.aiApiUrl}/api/ai/status`);
                this.status = data;
            } catch (err: any) {
                this.error = err.message || 'Error fetching status';
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
                this.error = err.message || 'Error fetching sentiment';
                console.error(`Failed to fetch AI Sentiment for ${pair}`, err);
            }
        },
        async fetchSignals(limit: number = 20) {
            try {
                const { data } = await axios.get<AISignal[]>(`${this.aiApiUrl}/api/ai/signals?limit=${limit}`);
                this.signals = data;
            } catch (err: any) {
                this.error = err.message || 'Error fetching signals';
                console.error('Failed to fetch AI Signals', err);
            }
        },
        async fetchCostSummary() {
            try {
                const { data } = await axios.get<AICostSummary>(`${this.aiApiUrl}/api/ai/cost`);
                this.costSummary = data;
            } catch (err: any) {
                this.error = err.message || 'Error fetching cost summary';
                console.error('Failed to fetch AI Cost', err);
            }
        },
        async fetchAutonomy() {
            try {
                const { data } = await axios.get<AIAutonomy>(`${this.aiApiUrl}/api/ai/autonomy`);
                this.autonomy = data;
            } catch (err: any) {
                this.error = err.message || 'Error fetching autonomy level';
                console.error('Failed to fetch AI Autonomy', err);
            }
        },
        async fetchRisk() {
            try {
                const { data } = await axios.get<AIRisk>(`${this.aiApiUrl}/api/ai/risk`);
                this.risk = data;
            } catch (err: any) {
                this.error = err.message || 'Error fetching risk info';
                console.error('Failed to fetch AI Risk', err);
            }
        },
        async fetchForgonePnl() {
            try {
                const { data } = await axios.get<AIForgonePnl>(`${this.aiApiUrl}/api/ai/forgone`);
                this.forgonePnl = data;
            } catch (err: any) {
                this.error = err.message || 'Error fetching forgone PNL';
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
                this.error = err.message || 'Error fetching confidence history';
                console.error('Failed to fetch Confidence History', err);
            }
        },
        async fetchHealth() {
            try {
                const { data } = await axios.get<AIHealth>(`${this.aiApiUrl}/api/ai/health`);
                this.health = data;
            } catch (err: any) {
                this.error = err.message || 'Error fetching health';
                console.error('Failed to fetch AI Health', err);
            }
        },
        async fetchMetrics() {
            try {
                const { data } = await axios.get<AIMetrics>(`${this.aiApiUrl}/api/ai/metrics`);
                this.metrics = data;
            } catch (err: any) {
                this.error = err.message || 'Error fetching metrics';
                console.error('Failed to fetch AI Metrics', err);
            }
        },
        async fetchAll() {
            this.loading = true;
            this.error = null;
            try {
                await Promise.all([
                    this.fetchStatus(),
                    this.fetchSignals(50),
                    this.fetchCostSummary(),
                    this.fetchAutonomy(),
                    this.fetchRisk(),
                    this.fetchForgonePnl()
                ]);
            } catch (err: any) {
                this.error = 'Error fetching multiple AI endpoints';
            } finally {
                this.loading = false;
            }
        }
    }
});
