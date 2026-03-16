<template>
  <div class="p-3 sm:p-4 md:p-6 max-w-[1600px] mx-auto space-y-4">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <h1 class="text-xl sm:text-2xl font-bold flex items-center gap-2">
        <i class="pi pi-bolt text-primary"></i> AI Trading
      </h1>
      <div class="flex items-center gap-3">
        <span v-if="aiStore.lastFetchTime" class="text-xs text-gray-500 dark:text-gray-400 hidden md:inline">
          {{ aiStore.lastFetchTime?.toLocaleTimeString() }}
        </span>
        <Tag :severity="aiStore.isAiOnline ? 'success' : 'danger'"
             :value="aiStore.isAiOnline ? 'ONLINE' : 'OFFLINE'" />
        <Button icon="pi pi-refresh" @click="refresh" :loading="aiStore.loading"
                class="p-button-sm p-button-text p-button-rounded" />
      </div>
    </div>

    <!-- Error / Offline / Stale banners -->
    <Message v-if="aiStore.error" severity="error" :closable="false" class="text-sm">
      <strong>{{ aiStore.error }}</strong>
      <Button label="Retry" icon="pi pi-refresh" @click="refresh" :loading="aiStore.loading"
              class="p-button-sm p-button-text ml-2" />
    </Message>
    <Message v-else-if="!aiStore.isAiOnline && aiStore.lastFetchTime" severity="warn" :closable="false" class="text-sm">
      AI Backend Offline — showing cached data from {{ aiStore.lastFetchTime?.toLocaleTimeString() }}
    </Message>
    <Message v-else-if="aiStore.isStaleData" severity="info" :closable="false" class="text-sm">
      Data is {{ staleMinutes }} minutes old.
      <Button label="Refresh" icon="pi pi-refresh" @click="refresh" :loading="aiStore.loading"
              class="p-button-sm p-button-text ml-2" />
    </Message>

    <!-- Row 1: Key Metrics (5 cards) -->
    <div v-if="aiStore.loading && !aiStore.lastFetchTime" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
      <Skeleton v-for="i in 5" :key="i" height="90px" class="rounded-lg" />
    </div>
    <div v-else class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">

      <!-- Daily P&L -->
      <div class="p-3 rounded-lg border dark:border-gray-700 bg-white dark:bg-gray-800">
        <div class="text-xs text-gray-600 dark:text-gray-400 mb-1">Today's P&L</div>
        <div class="text-lg sm:text-xl font-bold" :class="dailyPnlColor">
          {{ dailyPnlFormatted }}
        </div>
        <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
          {{ aiStore.dailyStats?.wins || 0 }}W / {{ aiStore.dailyStats?.losses || 0 }}L
          ({{ aiStore.dailyStats?.closed_today || 0 }} trades)
        </div>
      </div>

      <!-- Portfolio -->
      <div class="p-3 rounded-lg border dark:border-gray-700 bg-white dark:bg-gray-800">
        <div class="text-xs text-gray-600 dark:text-gray-400 mb-1">Portfolio</div>
        <div class="text-lg sm:text-xl font-bold">{{ aiStore.portfolioValueFormatted }}</div>
        <div class="text-xs text-gray-500 dark:text-gray-400 mt-1">
          {{ aiStore.portfolio?.stake_currency || 'USDT' }} |
          {{ Object.keys(aiStore.portfolio?.assets || {}).length }} asset
        </div>
      </div>

      <!-- Signal Accuracy (was: Win Rate) -->
      <div class="p-3 rounded-lg border dark:border-gray-700 bg-white dark:bg-gray-800">
        <div class="text-xs text-gray-600 dark:text-gray-400 mb-1">Signal Accuracy</div>
        <div class="text-lg sm:text-xl font-bold" :class="winRateColor">
          {{ aiStore.winRate.toFixed(0) }}%
        </div>
        <ProgressBar :value="aiStore.winRate" :showValue="false"
                     style="height: 4px;" class="mt-2" />
      </div>

      <!-- Daily Loss Limit (was: Risk Budget) -->
      <div class="p-3 rounded-lg border dark:border-gray-700 bg-white dark:bg-gray-800">
        <div class="text-xs text-gray-600 dark:text-gray-400 mb-1">Daily Loss Limit</div>
        <div class="text-lg sm:text-xl font-bold" :class="riskColor">
          {{ aiStore.risk?.utilization_pct?.toFixed(1) || '0' }}%
        </div>
        <ProgressBar :value="aiStore.risk?.utilization_pct || 0" :showValue="false"
                     style="height: 4px;" class="mt-2" />
      </div>

      <!-- Fear & Greed -->
      <div class="p-3 rounded-lg border dark:border-gray-700 bg-white dark:bg-gray-800">
        <div class="text-xs text-gray-600 dark:text-gray-400 mb-1">Fear & Greed</div>
        <div class="text-lg sm:text-xl font-bold" :class="fgColor">
          {{ aiStore.fearGreedIndex }}
        </div>
        <div class="text-xs mt-1" :class="fgColor">{{ fgLabel }}</div>
      </div>
    </div>

    <!-- $100 Hypothetical + Overall Confidence -->
    <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
      <div v-if="aiStore.hypothetical && aiStore.hypothetical.total_trades > 0"
           class="p-3 rounded-lg border dark:border-gray-700 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30">
        <div class="flex items-center justify-between">
          <div class="text-xs text-gray-600 dark:text-gray-400 font-semibold">$100 Simulation</div>
          <Tag severity="info" :value="`${aiStore.hypothetical.total_trades} trades`" class="text-xs" />
        </div>
        <div class="flex flex-col sm:flex-row items-baseline gap-2 mt-1">
          <span class="text-lg font-bold">${{ aiStore.hypothetical.current_balance.toFixed(2) }}</span>
          <span class="text-sm font-semibold" :class="aiStore.hypothetical.total_return_pct >= 0 ? 'text-green-500 dark:text-green-400' : 'text-red-500 dark:text-red-400'">
            {{ aiStore.hypothetical.total_return_pct >= 0 ? '+' : '' }}{{ aiStore.hypothetical.total_return_pct.toFixed(2) }}%
          </span>
        </div>
      </div>

      <!-- Overall AI Confidence (ConfidenceScore integration) -->
      <div v-if="recentSignals.length > 0"
           class="p-3 rounded-lg border dark:border-gray-700 bg-white dark:bg-gray-800 flex items-center gap-4">
        <ConfidenceScore :confidence="overallConfidence" />
        <div>
          <div class="text-xs text-gray-600 dark:text-gray-400">Avg Confidence</div>
          <div class="text-sm text-gray-500 dark:text-gray-400 mt-1">Last {{ recentSignals.length }} signals</div>
        </div>
      </div>
    </div>

    <!-- Status Bar -->
    <div class="flex flex-wrap gap-2 text-sm">
      <Tag severity="info" class="font-mono">
        <i class="pi pi-bolt mr-1"></i>
        L{{ aiStore.autonomy?.current_level ?? '?' }} {{ levelName }}
      </Tag>
      <Tag severity="secondary" class="font-mono">
        Size: {{ ((aiStore.autonomy?.kelly_fraction || 0) * 100).toFixed(0) }}%
      </Tag>
      <Tag severity="secondary" class="font-mono">
        {{ aiStore.risk?.active_positions || 0 }} positions
      </Tag>
      <Tag severity="secondary" class="font-mono">
        {{ aiStore.metrics?.total_decisions || 0 }} inferences (24h)
      </Tag>
      <Tag severity="secondary" class="font-mono">
        Cache: {{ ((aiStore.metrics?.cache_hit_rate || 0) * 100).toFixed(0) }}%
      </Tag>
      <Tag severity="secondary" class="font-mono">
        Cost: ${{ aiStore.costSummary?.today_cost?.toFixed(4) || '0.0000' }}
      </Tag>
      <Tag v-if="aiStore.status?.active_model" severity="secondary" class="font-mono">
        {{ aiStore.status.active_model }}
      </Tag>
    </div>

    <!-- Market Sentiment (SentimentDisplay integration) -->
    <div v-if="aiStore.marketSentiment && Object.keys(aiStore.marketSentiment.coins || {}).length > 0"
         class="border dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 overflow-hidden">
      <div class="p-3 border-b dark:border-gray-700">
        <h2 class="font-bold text-sm flex items-center gap-2">
          <i class="pi pi-globe"></i> Market Sentiment
        </h2>
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 p-3">
        <SentimentDisplay
          v-for="(data, coin) in topSentimentCoins" :key="coin"
          :pair="`${coin}/USDT`"
          :sentiment="{ pair: `${coin}/USDT`, sentiment_1h: data.sentiment_1h, sentiment_4h: data.sentiment_4h, sentiment_24h: data.sentiment_24h, fear_greed: aiStore.fearGreedIndex, source_count: data.news_count, last_update: '' }" />
      </div>
    </div>

    <!-- Alerts Feed -->
    <div v-if="aiStore.alerts && aiStore.alerts.length > 0" class="space-y-2">
      <Message v-for="(alert, idx) in aiStore.alerts.slice(0, 5)" :key="idx"
               :severity="alertSeverity(alert.level)" :closable="true" class="text-sm">
        {{ alert.message }}
        <span class="text-xs text-gray-500 dark:text-gray-400 ml-2">{{ formatTime(alert.timestamp) }}</span>
      </Message>
    </div>

    <!-- Blocked Signals Impact (was: Forgone P&L) -->
    <Message v-if="forgoneWarning" :severity="forgoneSeverity" :closable="false" class="text-sm">
      <strong>Blocked Signals: ${{ aiStore.forgonePnl?.total_forgone?.toFixed(2) }}</strong>
      {{ forgoneMessage }}
      <span class="text-xs ml-2">({{ aiStore.forgonePnl?.recent_signals || 0 }} signals tracked)</span>
    </Message>

    <!-- Recent Signals -->
    <div class="border dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 overflow-hidden">
      <div class="flex justify-between items-center p-3 border-b dark:border-gray-700">
        <h2 class="font-bold flex items-center gap-2">
          <i class="pi pi-chart-line"></i> Recent Signals
        </h2>
        <router-link to="/ai/analytics" class="text-xs text-primary hover:underline">
          View all &rarr;
        </router-link>
      </div>
      <DataTable :value="recentSignals" :rows="10" responsiveLayout="stack" breakpoint="640px"
                 class="p-datatable-sm" @row-click="onRowClick" rowHover
                 :loading="aiStore.loading">
        <Column field="pair" header="Pair">
          <template #body="{ data }">
            <span class="font-mono font-bold text-sm">{{ data.pair }}</span>
          </template>
        </Column>
        <Column field="signal" header="Signal">
          <template #body="{ data }">
            <Tag :severity="signalSeverity(data.signal)" :value="data.signal" />
          </template>
        </Column>
        <Column field="confidence" header="Confidence">
          <template #body="{ data }">
            <div class="flex items-center gap-2">
              <ProgressBar :value="data.confidence * 100" :showValue="false"
                           style="height: 6px; width: 50px;" />
              <span class="text-xs font-mono">{{ (data.confidence * 100).toFixed(0) }}%</span>
            </div>
          </template>
        </Column>
        <Column field="outcome" header="Outcome">
          <template #body="{ data }">
            <span :class="outcomeClass(data.outcome)" class="font-mono text-sm">
              {{ data.outcome || 'Pending' }}
            </span>
          </template>
        </Column>
        <Column field="timestamp" header="Time">
          <template #body="{ data }">
            <span class="text-xs text-gray-500 dark:text-gray-400">{{ formatTime(data.timestamp) }}</span>
          </template>
        </Column>
      </DataTable>
      <div v-if="!recentSignals.length && !aiStore.loading"
           class="p-6 text-center text-gray-500 dark:text-gray-400 text-sm">
        No signals yet. AI pipeline is warming up...
      </div>
    </div>

    <!-- Signal Reasoning Modal -->
    <Dialog v-model:visible="showModal" header="AI Reasoning"
            :style="{width: '95vw', maxWidth: '800px'}" modal closable>
      <TradeReasoning v-if="selectedSignal" :signal="selectedSignal" />
    </Dialog>

    <!-- Quick Links -->
    <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
      <router-link to="/ai/analytics"
        class="p-3 rounded-lg border dark:border-gray-700 bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 text-center text-sm font-medium transition-colors">
        <i class="pi pi-chart-bar mr-1"></i> Analytics
      </router-link>
      <router-link to="/ai/risk"
        class="p-3 rounded-lg border dark:border-gray-700 bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 text-center text-sm font-medium transition-colors">
        <i class="pi pi-shield mr-1"></i> Risk
      </router-link>
      <router-link to="/ai/settings"
        class="p-3 rounded-lg border dark:border-gray-700 bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 text-center text-sm font-medium transition-colors">
        <i class="pi pi-cog mr-1"></i> Settings
      </router-link>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, onMounted } from 'vue';
import { useAiStore } from '@/stores/aiStore';
import type { AISignal } from '@/stores/aiStore';

import Tag from 'primevue/tag';
import Button from 'primevue/button';
import ProgressBar from 'primevue/progressbar';
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
import Message from 'primevue/message';
import Dialog from 'primevue/dialog';
import Skeleton from 'primevue/skeleton';
import TradeReasoning from './TradeReasoning.vue';
import ConfidenceScore from './ConfidenceScore.vue';
import SentimentDisplay from './SentimentDisplay.vue';

const aiStore = useAiStore();

const showModal = ref(false);
const selectedSignal = ref<AISignal | null>(null);

onMounted(async () => {
  if (!aiStore.lastFetchTime) {
    await aiStore.fetchAll();
  }
});

const refresh = async () => {
  await aiStore.fetchAll();
};

// --- Computed ---

const recentSignals = computed(() => (aiStore.signals || []).slice(0, 15));

const overallConfidence = computed(() => {
  const sigs = aiStore.signals || [];
  if (sigs.length === 0) return 0;
  const recent = sigs.slice(0, 10);
  return recent.reduce((sum, s) => sum + s.confidence, 0) / recent.length;
});

const topSentimentCoins = computed(() => {
  const coins = aiStore.marketSentiment?.coins || {};
  const entries = Object.entries(coins).slice(0, 3);
  return Object.fromEntries(entries);
});

const levelName = computed(() => {
  const names = ['Nano-live', 'Micro-live', 'Small-live', 'Cautious-live', 'Standard-live', 'Full-auto'];
  return names[aiStore.autonomy?.current_level ?? 0] || '';
});

const dailyPnlFormatted = computed(() => {
  const pnl = aiStore.dailyStats?.daily_pnl ?? 0;
  const pct = aiStore.dailyStats?.daily_pnl_pct ?? 0;
  const sign = pnl >= 0 ? '+' : '';
  return `${sign}$${pnl.toFixed(2)} (${sign}${pct.toFixed(2)}%)`;
});

const dailyPnlColor = computed(() => {
  const pnl = aiStore.dailyStats?.daily_pnl ?? 0;
  if (pnl > 0) return 'text-green-500 dark:text-green-400';
  if (pnl < 0) return 'text-red-500 dark:text-red-400';
  return 'text-gray-500 dark:text-gray-400';
});

const winRateColor = computed(() => {
  const wr = aiStore.winRate;
  if (wr >= 60) return 'text-green-500 dark:text-green-400';
  if (wr >= 45) return 'text-yellow-500 dark:text-yellow-400';
  return 'text-red-500 dark:text-red-400';
});

const riskColor = computed(() => {
  const pct = aiStore.risk?.utilization_pct || 0;
  if (pct >= 100) return 'text-red-500 dark:text-red-400';
  if (pct >= 75) return 'text-orange-500 dark:text-orange-400';
  if (pct >= 50) return 'text-yellow-500 dark:text-yellow-400';
  return 'text-green-500 dark:text-green-400';
});

const fgColor = computed(() => {
  const fg = aiStore.fearGreedIndex;
  if (fg >= 75) return 'text-green-500 dark:text-green-400';
  if (fg >= 55) return 'text-green-400 dark:text-green-300';
  if (fg >= 45) return 'text-yellow-500 dark:text-yellow-400';
  if (fg >= 25) return 'text-orange-500 dark:text-orange-400';
  return 'text-red-500 dark:text-red-400';
});

const fgLabel = computed(() => {
  const fg = aiStore.fearGreedIndex;
  if (fg >= 75) return 'Extreme Greed';
  if (fg >= 55) return 'Greed';
  if (fg >= 45) return 'Neutral';
  if (fg >= 25) return 'Fear';
  return 'Extreme Fear';
});

const staleMinutes = computed(() => {
  if (!aiStore.lastFetchTime) return 0;
  return Math.floor((Date.now() - aiStore.lastFetchTime.getTime()) / 60000);
});

const forgoneWarning = computed(() => {
  const f = aiStore.forgonePnl?.total_forgone;
  return f !== undefined && f !== null && Math.abs(f) > 0.01;
});

const forgoneSeverity = computed(() => {
  const f = aiStore.forgonePnl?.total_forgone || 0;
  if (f > 2) return 'error';
  if (f > 0) return 'warn';
  return 'success';
});

const forgoneMessage = computed(() => {
  const f = aiStore.forgonePnl?.total_forgone || 0;
  if (f > 0) return '- Guardrails cost you money. Consider raising autonomy level.';
  return '- Guardrails saved you from losses.';
});

// --- Methods ---

function alertSeverity(level: string) {
  if (level === 'ERROR' || level === 'CRITICAL') return 'error';
  if (level === 'WARNING') return 'warn';
  return 'info';
}

function signalSeverity(signal: string) {
  if (signal === 'BULLISH') return 'success';
  if (signal === 'BEARISH') return 'danger';
  return 'warn';
}

function outcomeClass(outcome?: string) {
  if (!outcome || outcome === 'Pending') return 'text-gray-400 dark:text-gray-500';
  const val = parseFloat(outcome.replace('%', ''));
  if (val > 0) return 'text-green-500 dark:text-green-400 font-bold';
  if (val < 0) return 'text-red-500 dark:text-red-400 font-bold';
  return 'text-gray-500 dark:text-gray-400';
}

function formatTime(ts: string) {
  if (!ts) return '-';
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
}

function onRowClick(event: any) {
  selectedSignal.value = event.data as AISignal;
  showModal.value = true;
}
</script>
