import { describe, expect, it } from 'vitest';

import { usePlotTemplates } from '@/composables/plotTemplates';
import type { PlotConfig } from '@/types';

describe('replaceTemplateColumns', () => {
  it('Updates main plot values', () => {
    const { replaceTemplateColumns } = usePlotTemplates();
    const reMapping = { ema: 'ema_14' };
    const template: Partial<PlotConfig> = {
      main_plot: {
        ema: {
          color: '#ff8000',
          type: 'line',
        },
      },
    };
    const expected: Partial<PlotConfig> = {
      main_plot: {
        ema_14: {
          color: '#ff8000',
          type: 'line',
        },
      },
    };
    expect(replaceTemplateColumns(template, reMapping)).toEqual(expected);
  });
  it('Updates subplot plot values', () => {
    const { replaceTemplateColumns } = usePlotTemplates();
    const reMapping = { rsi: 'rsi_14' };
    const template: Partial<PlotConfig> = {
      subplots: {
        RSI: {
          rsi: {
            color: '#ff8000',
            type: 'line',
          },
        },
      },
    };
    const expected: Partial<PlotConfig> = {
      subplots: {
        RSI: {
          rsi_14: {
            color: '#ff8000',
            type: 'line',
          },
        },
      },
    };
    expect(replaceTemplateColumns(template, reMapping)).toEqual(expected);
  });
  it('Updates both main and subplot values', () => {
    const { replaceTemplateColumns } = usePlotTemplates();
    const reMapping = { ema: 'ema_200', macd: 'macd_5m', macdsignal: 'macdsignal_5m' };
    const template: Partial<PlotConfig> = {
      main_plot: {
        ema: {
          color: '#ff8000',
          type: 'line',
        },
      },
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
    };
    const expected: Partial<PlotConfig> = {
      main_plot: {
        ema_200: {
          color: '#ff8000',
          type: 'line',
        },
      },
      subplots: {
        MACD: {
          macdsignal_5m: {
            color: '#ff8000',
            type: 'line',
          },
          macd_5m: {
            color: '#1370f4',
            type: 'line',
          },
        },
      },
    };
    expect(replaceTemplateColumns(template, reMapping)).toEqual(expected);
  });

  it('Updates "fill to" values', () => {
    const { replaceTemplateColumns } = usePlotTemplates();
    const reMapping = {
      bb_upperband: 'upperband',
      bb_lowerband: 'lowerband',
      macd: 'macd_5m',
    };
    const template: Partial<PlotConfig> = {
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
      subplots: {
        MACD: {
          macdsignal: {
            color: '#ff8000',
            type: 'line',
            fill_to: 'macd',
          },
          macd: {
            color: '#1370f4',
            type: 'line',
          },
        },
      },
    };
    const expected: Partial<PlotConfig> = {
      main_plot: {
        upperband: {
          color: '#008af4',
          type: 'line',
          fill_to: 'lowerband',
        },
        lowerband: {
          color: '#008af4',
          type: 'line',
        },
      },
      subplots: {
        MACD: {
          macdsignal: {
            color: '#ff8000',
            type: 'line',
            fill_to: 'macd_5m',
          },
          macd_5m: {
            color: '#1370f4',
            type: 'line',
          },
        },
      },
    };
    expect(replaceTemplateColumns(template, reMapping)).toEqual(expected);
  });
});

describe('applyPlotTemplate', () => {
  it('Updates main plot values', () => {
    const { applyPlotTemplate } = usePlotTemplates();
    const reMapping = { rsi: 'rsi_14' };

    const currentConfig: PlotConfig = {
      main_plot: { ema: { color: '#ff8000', type: 'line' } },
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
    };
    const expected: Partial<PlotConfig> = {
      main_plot: { ema: { color: '#ff8000', type: 'line' } },
      subplots: {
        RSI: {
          rsi_14: {
            color: '#ff8000',
            type: 'line',
          },
        },
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
    };
    expect(applyPlotTemplate('RSI', currentConfig, reMapping)).toEqual(expected);
  });
});

describe('replaceTemplateColumns - validation', () => {
  it('handles invalid values in main_plot', () => {
    const { replaceTemplateColumns } = usePlotTemplates();
    const template = {
      main_plot: {
        valid: { color: 'red' },
        invalid: 'not an object' as any,
        nullValue: null as any,
      },
    };
    const result = replaceTemplateColumns(template, { valid: 'valid_mapped' });
    expect(result.main_plot).toEqual({
      valid_mapped: { color: 'red' },
    });
    expect(result.main_plot?.invalid).toBeUndefined();
    expect(result.main_plot?.nullValue).toBeUndefined();
  });

  it('handles invalid values in subplots', () => {
    const { replaceTemplateColumns } = usePlotTemplates();
    const template = {
      subplots: {
        MACD: {
          valid: { color: 'blue' },
          invalid: 123 as any,
        },
      },
    };
    const result = replaceTemplateColumns(template, { valid: 'valid_mapped' });
    expect(result.subplots?.MACD).toEqual({
      valid_mapped: { color: 'blue' },
    });
    expect(result.subplots?.MACD.invalid).toBeUndefined();
  });

  it('handles missing main_plot or subplots', () => {
    const { replaceTemplateColumns } = usePlotTemplates();
    expect(replaceTemplateColumns({}, {})).toEqual({});
    expect(replaceTemplateColumns({ main_plot: {} }, {})).toEqual({ main_plot: {} });
    expect(replaceTemplateColumns({ subplots: {} }, {})).toEqual({ subplots: {} });
  });
});
