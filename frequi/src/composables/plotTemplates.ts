import { type IndicatorConfig, isIndicatorConfig, type PlotConfig, type PlotConfigTemplate } from '@/types';

const plotTemplates = ref<PlotConfigTemplate>({
  BollingerBands: {
    main_plot: {
      bb_upperband: {
        color: '#008af4',
        type: 'line',
        fill_to: 'bb_lowerband',
      },
      bb_lowerband: {
        color: '#008af4',
        type: 'line',
      },
    },
  },
  RSI: {
    subplots: {
      RSI: {
        rsi: {
          color: '#ff8000',
          type: 'line',
        },
      },
    },
  },
  MACD: {
    subplots: {
      MACD: {
        macdsignal: {
          color: '#ff8000',
          type: 'line',
        },
        macd: {
          color: '#1370f4',
          type: 'line',
        },
      },
    },
  },
});

function replaceTemplateColumns(template: Partial<PlotConfig>, nameMap: Record<string, string>) {
  // Replace the keys of all elements in main_plot
  const newTemplate = deepClone(template);
  if (!nameMap) {
    return newTemplate;
  }

  if (template.main_plot) {
    const newMainPlot: Record<string, IndicatorConfig> = {};
    for (const [key, val] of Object.entries(template.main_plot)) {
      const newKey = nameMap[key] || key;
      if (isIndicatorConfig(val)) {
        newMainPlot[newKey] = { ...val };
        if (newMainPlot[newKey].fill_to !== undefined) {
          newMainPlot[newKey].fill_to =
            nameMap[newMainPlot[newKey].fill_to] || newMainPlot[newKey].fill_to;
        }
      }
    }
    newTemplate.main_plot = newMainPlot;
  }

  // Replace the keys of all elements in subplots
  if (template.subplots) {
    const newSubplots: Record<string, Record<string, IndicatorConfig>> = {};
    for (const [subplotKey, subplotValue] of Object.entries(template.subplots)) {
      const newSubplot: Record<string, IndicatorConfig> = {};
      for (const [key, val] of Object.entries(subplotValue)) {
        const newKey = nameMap[key] || key;
        if (isIndicatorConfig(val)) {
          newSubplot[newKey] = { ...val };
          if (newSubplot[newKey].fill_to !== undefined) {
            newSubplot[newKey].fill_to =
              nameMap[newSubplot[newKey].fill_to] || newSubplot[newKey].fill_to;
          }
        }
      }
      newSubplots[subplotKey] = newSubplot;
    }
    newTemplate.subplots = newSubplots;
  }
  return newTemplate;
}

export function usePlotTemplates() {
  function getTemplateContent(templateName: string) {
    return plotTemplates.value[templateName] || {};
  }

  function applyPlotTemplate(
    templateName: string,
    currentConfig: PlotConfig,
    nameMap: Record<string, string> = {},
  ) {
    const template = getTemplateContent(templateName);
    if (!template) {
      return currentConfig;
    }
    const newTemplate = replaceTemplateColumns(template, nameMap);
    return deepMerge(currentConfig, newTemplate);
  }

  return {
    plotTemplates,
    applyPlotTemplate,
    getTemplateContent,
    replaceTemplateColumns,
    plotTemplateNames: computed(() => Object.keys(plotTemplates.value)),
  };
}
