<template>
  <div class="p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border dark:border-gray-700">
    <h3 class="text-sm font-bold mb-3 flex items-center justify-between">
      <span>Market Sentiment</span>
      <span class="text-xs font-normal text-gray-600 dark:text-gray-400">{{ pair }}</span>
    </h3>
    
    <div v-if="sentiment" class="space-y-4">
      <!-- Fear & Greed -->
      <div>
        <div class="flex justify-between text-xs mb-1">
          <span class="text-gray-600 dark:text-gray-400">Fear & Greed Index</span>
          <span class="font-bold" :class="getFgColor(sentiment.fear_greed)">{{ sentiment.fear_greed }}/100</span>
        </div>
        <ProgressBar :value="sentiment.fear_greed" :showValue="false" style="height: 6px;" :class="getFgBgColor(sentiment.fear_greed)" />
        <div class="text-xs text-right mt-1 text-gray-400">{{ getFgLabel(sentiment.fear_greed) }}</div>
      </div>
      
      <!-- Multi-Timeframe -->
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-2 pt-2 border-t dark:border-gray-700">
        <div class="text-center">
          <div class="text-xs text-gray-600 dark:text-gray-400">1H</div>
          <div class="font-bold text-sm" :class="getSentimentColor(sentiment.sentiment_1h)">{{ formatScore(sentiment.sentiment_1h) }}</div>
        </div>
        <div class="text-center">
          <div class="text-xs text-gray-600 dark:text-gray-400">4H</div>
          <div class="font-bold text-sm" :class="getSentimentColor(sentiment.sentiment_4h)">{{ formatScore(sentiment.sentiment_4h) }}</div>
        </div>
        <div class="text-center">
          <div class="text-xs text-gray-600 dark:text-gray-400">24H</div>
          <div class="font-bold text-sm" :class="getSentimentColor(sentiment.sentiment_24h)">{{ formatScore(sentiment.sentiment_24h) }}</div>
        </div>
      </div>
    </div>
    <div v-else class="text-sm text-gray-400 text-center py-4">No data available</div>
  </div>
</template>

<script setup lang="ts">
import type { AISentiment } from '@/stores/aiStore';
import ProgressBar from 'primevue/progressbar';

defineProps<{
  pair: string,
  sentiment: AISentiment | null
}>();

const formatScore = (val: number) => {
  return val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2);
};

const getSentimentColor = (val: number) => {
  if (val > 0.5) return 'text-green-500';
  if (val < -0.5) return 'text-red-500';
  return 'text-gray-400';
};

const getFgColor = (val: number) => {
  if (val >= 75) return 'text-green-500';
  if (val >= 55) return 'text-green-400';
  if (val >= 45) return 'text-yellow-500';
  if (val >= 25) return 'text-orange-500';
  return 'text-red-500';
};

const getFgBgColor = (val: number) => {
  if (val >= 75) return 'bg-green-500';
  if (val >= 55) return 'bg-green-400';
  if (val >= 45) return 'bg-yellow-500';
  if (val >= 25) return 'bg-orange-500';
  return 'bg-red-500';
};

const getFgLabel = (val: number) => {
  if (val >= 75) return 'Extreme Greed';
  if (val >= 55) return 'Greed';
  if (val >= 45) return 'Neutral';
  if (val >= 25) return 'Fear';
  return 'Extreme Fear';
};
</script>
