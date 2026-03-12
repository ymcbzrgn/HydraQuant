<template>
  <div class="card p-4">
    <div class="flex justify-between items-center mb-4">
      <h2 class="text-xl font-bold">AI Trade Signals</h2>
      <div class="flex gap-2">
        <InputText v-model="filters.pair" placeholder="Search Pair..." class="p-inputtext-sm" />
        <Dropdown v-model="filters.direction" :options="['ALL', 'BULLISH', 'BEARISH', 'NEUTRAL']" class="p-dropdown-sm w-32" />
        <Button icon="pi pi-refresh" @click="refreshData" :loading="aiStore.loading" class="p-button-sm p-button-outlined" />
      </div>
    </div>
    
    <Message v-if="aiStore.error" severity="error" :closable="false">{{ aiStore.error }}</Message>

    <DataTable :value="filteredSignals" :paginator="true" :rows="10" 
               responsiveLayout="scroll" class="p-datatable-sm w-full mt-2" 
               :loading="aiStore.loading" @row-click="onRowClick" rowHover>
      
      <Column field="pair" header="Pair" :sortable="true"></Column>
      
      <Column field="signal" header="Direction" :sortable="true">
        <template #body="slotProps">
          <Tag :severity="getDirectionSeverity(slotProps.data.signal)" :value="slotProps.data.signal"></Tag>
        </template>
      </Column>
      
      <Column field="confidence" header="Confidence" :sortable="true">
        <template #body="slotProps">
          <div class="flex items-center gap-2">
            <ProgressBar :value="slotProps.data.confidence * 100" :showValue="false" 
                         style="height: 6px; width: 60px;" :class="getConfidenceColor(slotProps.data.confidence)" />
            <span class="text-xs">{{ (slotProps.data.confidence * 100).toFixed(0) }}%</span>
          </div>
        </template>
      </Column>
      
      <Column field="timestamp" header="Time" :sortable="true">
        <template #body="slotProps">
          {{ formatDate(slotProps.data.timestamp) }}
        </template>
      </Column>
      
      <Column field="outcome" header="Outcome" :sortable="true">
        <template #body="slotProps">
          <span :class="{'text-green-500 font-bold': parseOutcome(slotProps.data.outcome) > 0, 
                         'text-red-500 font-bold': parseOutcome(slotProps.data.outcome) < 0}">
             {{ slotProps.data.outcome }}
          </span>
        </template>
      </Column>
    </DataTable>

    <!-- Modal for Reasoning -->
    <Dialog v-model:visible="showModal" header="AI Reasoning" :style="{width: '60vw'}" modal closable>
      <TradeReasoning v-if="selectedSignal" :signal="selectedSignal" />
    </Dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue';
import { useAiStore } from '@/stores/aiStore';
import type { AISignal } from '@/stores/aiStore';
import TradeReasoning from './TradeReasoning.vue';

// PrimeVue components
import DataTable from 'primevue/datatable';
import Column from 'primevue/column';
import Tag from 'primevue/tag';
import ProgressBar from 'primevue/progressbar';
import InputText from 'primevue/inputtext';
import Dropdown from 'primevue/dropdown';
import Button from 'primevue/button';
import Message from 'primevue/message';
import Dialog from 'primevue/dialog';

const aiStore = useAiStore();
const refreshTimer = ref<number | null>(null);

const filters = ref({
  pair: '',
  direction: 'ALL'
});

const showModal = ref(false);
const selectedSignal = ref<AISignal | null>(null);

const filteredSignals = computed(() => {
  return aiStore.signals.filter(s => {
    const matchPair = s.pair.toLowerCase().includes(filters.value.pair.toLowerCase());
    const matchDir = filters.value.direction === 'ALL' || s.signal === filters.value.direction;
    return matchPair && matchDir;
  });
});

const refreshData = async () => {
  await aiStore.fetchSignals(50);
};

onMounted(async () => {
  await refreshData();
  refreshTimer.value = window.setInterval(refreshData, 60000); // 60s
});

onUnmounted(() => {
  if (refreshTimer.value) clearInterval(refreshTimer.value);
});

const onRowClick = (event: any) => {
  selectedSignal.value = event.data as AISignal;
  showModal.value = true;
};

const getDirectionSeverity = (signal: string) => {
  if (signal === 'BULLISH') return 'success';
  if (signal === 'BEARISH') return 'danger';
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
