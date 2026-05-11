export enum ChartType {
  line = 'line',
  bar = 'bar',
  scatter = 'scatter',
}

export type ChartTypeString = keyof typeof ChartType;

export interface IndicatorConfig {
  color?: string;
  type?: ChartType | ChartTypeString;
  fill_to?: string;
  scatterSymbolSize?: number;
}

export interface PlotConfig {
  main_plot: Record<string, IndicatorConfig>;
  subplots: Record<string, Record<string, IndicatorConfig>>;
  options?: {
    showTags?: boolean;
    markAreaZIndex?: number;
  };
}

export interface PlotConfigStorage {
  [key: string]: PlotConfig;
}

export interface PlotConfigTemplate {
  [key: string]: Partial<PlotConfig>;
}

export const EMPTY_PLOTCONFIG: PlotConfig = { main_plot: {}, subplots: {} };

export function isIndicatorConfig(obj: any): obj is IndicatorConfig {
  if (typeof obj !== 'object' || obj === null) {
    return false;
  }

  if ('color' in obj && typeof obj.color !== 'string' && obj.color !== undefined) return false;
  if ('type' in obj && typeof obj.type !== 'string' && obj.type !== undefined) return false;
  if ('fill_to' in obj && typeof obj.fill_to !== 'string' && obj.fill_to !== undefined) return false;
  if ('scatterSymbolSize' in obj && typeof obj.scatterSymbolSize !== 'number' && obj.scatterSymbolSize !== undefined) return false;

  return true;
}
