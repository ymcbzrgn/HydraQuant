<template>
  <div class="p-4" v-if="signal">
    <div class="flex items-center gap-4 mb-4">
      <div class="text-2xl font-bold">{{ signal.pair }}</div>
      <Tag :severity="getDirectionSeverity(signal.signal)" :value="signal.signal" class="text-lg"></Tag>
      <div class="ml-auto text-sm text-gray-500">{{ formatDate(signal.timestamp) }}</div>
    </div>
    
    <TabView>
      <TabPanel header="Structural Analysis">
        <div class="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg whitespace-pre-wrap leading-relaxed border dark:border-gray-700 text-sm">
          {{ signal.reasoning || 'No reasoning provided by the AI for this decision.' }}
        </div>
      </TabPanel>
      
      <TabPanel header="Parameters">
        <div class="grid grid-cols-2 gap-4">
          <div class="p-3 border rounded-lg dark:border-gray-700">
            <div class="text-sm text-gray-500 mb-1">Confidence Score</div>
            <div class="text-lg font-semibold flex items-center gap-2">
              <ProgressBar :value="signal.confidence * 100" :showValue="false" 
                           style="height: 8px; width: 100px;" :class="getConfidenceColor(signal.confidence)" />
              {{ (signal.confidence * 100).toFixed(1) }}%
            </div>
          </div>
          <div class="p-3 border rounded-lg dark:border-gray-700">
            <div class="text-sm text-gray-500 mb-1">Trade Outcome</div>
            <div class="text-lg font-semibold" 
                 :class="{'text-green-500': parseOutcome(signal.outcome) > 0, 
                          'text-red-500': parseOutcome(signal.outcome) < 0}">
              {{ signal.outcome || 'Pending' }}
            </div>
          </div>
        </div>
      </TabPanel>
    </TabView>
  </div>
</template>

<script setup lang="ts">
import { defineProps } from 'vue';
import type { AISignal } from '@/stores/aiStore';

import Tag from 'primevue/tag';
import TabView from 'primevue/tabview';
import TabPanel from 'primevue/tabpanel';
import ProgressBar from 'primevue/progressbar';

const props = defineProps<{
  signal: AISignal
}>();

const getDirectionSeverity = (signal: string) => {
  if (signal === 'BULL') return 'success';
  if (signal === 'BEAR') return 'danger';
  return 'warning';
};

const getConfidenceColor = (conf: number) => {
  if (conf > 0.85) return 'bg-green-500';
  if (conf > 0.70) return 'bg-green-400';
  if (conf > 0.50) return 'bg-yellow-400';
  if (conf > 0.35) return 'bg-orange-500';
  return 'bg-red-500';
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
