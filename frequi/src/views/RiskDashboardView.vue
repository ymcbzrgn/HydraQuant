<template>
  <div class="p-6 max-w-7xl mx-auto space-y-6">
    <div class="flex items-center justify-between">
      <h1 class="text-3xl font-bold flex items-center gap-3">
        <i class="pi pi-shield text-primary"></i> Risk Oversight Dashboard
      </h1>
      <Tag severity="danger" value="Live Market Constraints"></Tag>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      
      <!-- Primary Risk Controls -->
      <div class="space-y-6">
        <RiskPanel />
        <AutonomyLevel />
      </div>

      <!-- Secondary Risk Distributions -->
      <div class="space-y-6">
        <div class="card p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border dark:border-gray-700">
          <h2 class="text-xl font-bold mb-4">Position Allocation Risk</h2>
          
          <div class="flex flex-col gap-4">
            <!-- Simulated Positions since actual pairs aren't tracked historically per risk endpoint yet -->
            <div class="flex items-center justify-between p-3 border dark:border-gray-700 bg-gray-50 dark:bg-gray-900 rounded">
              <div class="flex items-center gap-3">
                <i class="pi pi-bitcoin text-orange-500 text-xl"></i>
                <div>
                  <div class="font-bold">BTC/USDT</div>
                  <div class="text-xs text-gray-500">Long • 2.5x Leverage</div>
                </div>
              </div>
              <div class="text-right">
                <div class="font-bold text-red-500">$500.00 at Risk</div>
                <div class="text-xs text-gray-400">2.5% VaR Contribution</div>
              </div>
            </div>

            <div class="flex items-center justify-between p-3 border dark:border-gray-700 bg-gray-50 dark:bg-gray-900 rounded">
              <div class="flex items-center gap-3">
                <i class="pi pi-ethereum text-blue-500 text-xl"></i>
                <div>
                  <div class="font-bold">ETH/USDT</div>
                  <div class="text-xs text-gray-500">Short • 1.0x Leverage</div>
                </div>
              </div>
              <div class="text-right">
                <div class="font-bold text-red-500">$250.00 at Risk</div>
                <div class="text-xs text-gray-400">1.2% VaR Contribution</div>
              </div>
            </div>

            <div class="flex items-center justify-between p-3 border dark:border-gray-700 bg-gray-50 dark:bg-gray-900 rounded opacity-50">
              <div class="flex items-center gap-3">
                <i class="pi pi-bolt text-yellow-500 text-xl"></i>
                <div>
                  <div class="font-bold">SOL/USDT</div>
                  <div class="text-xs text-gray-500">Pending Execution</div>
                </div>
              </div>
              <div class="text-right">
                <div class="font-bold text-orange-500">$850.00 Allocated</div>
                <div class="text-xs text-gray-400">Awaiting Autonomy Clearance</div>
              </div>
            </div>

          </div>
        </div>

        <div class="card p-4 bg-white dark:bg-gray-800 rounded-lg shadow-sm border dark:border-gray-700">
          <h2 class="text-xl font-bold mb-4">30-Day Drawdown Threshold</h2>
          
          <!-- Mock Chart visualization -->
          <div class="h-48 w-full border-b border-l border-gray-200 dark:border-gray-700 relative p-4 pl-0">
            <!-- Simulated safety lines -->
            <div class="absolute w-full border-t border-dashed border-red-500 top-1/2 opacity-50 z-0"></div>
            <div class="absolute text-[10px] text-red-500 right-2 top-[45%] z-0">Max Allowed (-15%)</div>
            
            <div class="absolute w-full border-t border-dashed border-orange-500 top-3/4 opacity-50 z-0"></div>
            <div class="absolute text-[10px] text-orange-500 right-2 top-[70%] z-0">Warning Zone (-10%)</div>

            <div class="w-full h-full flex items-end gap-1 relative z-10 px-8">
              <div v-for="(val, index) in mockDrawdowns" :key="index"
                   class="flex-1 bg-green-500 border-t"
                   :class="{'bg-red-500': val < -10, 'bg-orange-500': val < -5 && val >= -10}"
                   :style="{ height: `${100 + val}%` }"
                   :title="`DD: ${val.toFixed(2)}%`">
              </div>
            </div>
          </div>
          <div class="text-center text-xs text-gray-400 mt-2">Historical Drawdown %</div>

        </div>

      </div>

    </div>
  </div>
</template>

<script setup lang="ts">
import RiskPanel from '@/components/ai/RiskPanel.vue';
import AutonomyLevel from '@/components/ai/AutonomyLevel.vue';

import Tag from 'primevue/tag';

// Mock values simulating a stable baseline curving down into drawdown briefly 
const mockDrawdowns = [
  -1, -1.2, -0.8, -2.5, -4.0, -4.5, -8.2, -11.0, -12.5, -9.0, 
  -7.5, -5.0, -4.2, -3.1, -2.0, -1.5, -0.5, 0, 0, -1.0, 
  -0.5, -0.2, -2.0, -2.5, -1.0, -0.5, 0, -0.2, 0, 0
];
</script>
