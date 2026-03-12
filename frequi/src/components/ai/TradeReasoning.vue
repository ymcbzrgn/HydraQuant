<template>
  <div class="p-4" v-if="signal">
    <div class="flex items-center gap-4 mb-4 flex-wrap break-words">
      <div class="text-2xl font-bold">{{ signal.pair }}</div>
      <Tag :severity="getDirectionSeverity(signal.signal)" :value="signal.signal" class="text-lg"></Tag>
      <div class="ml-auto text-sm text-gray-500">{{ formatDate(signal.timestamp) }}</div>
    </div>

    <TabView>
      <TabPanel header="Structural Analysis">
        <div class="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg whitespace-pre-wrap leading-relaxed border dark:border-gray-700 text-sm break-words">
          {{ signal.reasoning || 'No reasoning provided by the AI for this decision.' }}
        </div>
      </TabPanel>

      <TabPanel header="Parameters">
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div class="p-3 border rounded-lg dark:border-gray-700 bg-white dark:bg-gray-900 flex flex-col items-center">
            <div class="text-sm text-gray-500 mb-2">Confidence Score</div>
            <ConfidenceScore :confidence="signal.confidence" />
          </div>
          <div class="p-3 border rounded-lg dark:border-gray-700 bg-white dark:bg-gray-900">
            <div class="text-sm text-gray-500 mb-1">Trade Outcome</div>
            <div class="text-lg font-semibold"
                 :class="{'text-green-500': parseOutcome(signal.outcome) > 0,
                          'text-red-500': parseOutcome(signal.outcome) < 0}">
              {{ signal.outcome || 'Pending' }}
            </div>
            <div v-if="signal.outcome && signal.outcome !== 'Pending'" class="text-xs text-gray-400 mt-2">
              Predicted {{ (signal.confidence * 100).toFixed(0) }}%
              &rarr; {{ parseOutcome(signal.outcome) > 0 ? 'Won' : 'Lost' }}
            </div>
          </div>
        </div>
      </TabPanel>

      <TabPanel header="Market Context">
        <SentimentDisplay :pair="signal.pair" :sentiment="pairSentiment" />
      </TabPanel>
    </TabView>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, watch } from 'vue';
import { useAiStore } from '@/stores/aiStore';
import type { AISignal, AISentiment } from '@/stores/aiStore';

import Tag from 'primevue/tag';
import TabView from 'primevue/tabview';
import TabPanel from 'primevue/tabpanel';
import ConfidenceScore from './ConfidenceScore.vue';
import SentimentDisplay from './SentimentDisplay.vue';

const aiStore = useAiStore();

const props = defineProps<{
  signal: AISignal
}>();

// Fetch pair-specific sentiment when signal changes
const fetchSentiment = () => {
  if (props.signal?.pair) {
    aiStore.fetchSentiment(props.signal.pair);
  }
};

onMounted(fetchSentiment);
watch(() => props.signal?.pair, fetchSentiment);

const pairSentiment = computed<AISentiment | null>(() => {
  if (!props.signal?.pair) return null;
  return aiStore.sentiment[props.signal.pair] || null;
});

const getDirectionSeverity = (signal: string) => {
  if (signal === 'BULLISH') return 'success';
  if (signal === 'BEARISH') return 'danger';
  return 'warning';
};

const formatDate = (isoStr: string) => {
  if (!isoStr) return '';
  return new Date(isoStr).toLocaleString();
};

const parseOutcome = (str?: string) => {
  if (!str || str === 'Pending') return 0;
  return parseFloat(str.replace('%', ''));
};
</script>
