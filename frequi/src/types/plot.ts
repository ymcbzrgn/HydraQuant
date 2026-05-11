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

export function isIndicatorConfig(value: unknown): value is IndicatorConfig {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    (typeof (value as IndicatorConfig).color === 'string' ||
      (value as IndicatorConfig).color === undefined) &&
    (typeof (value as IndicatorConfig).type === 'string' ||
      (value as IndicatorConfig).type === undefined) &&
    (typeof (value as IndicatorConfig).fill_to === 'string' ||
      (value as IndicatorConfig).fill_to === undefined) &&
    (typeof (value as IndicatorConfig).scatterSymbolSize === 'number' ||
      (value as IndicatorConfig).scatterSymbolSize === undefined)
  );
}

export const EMPTY_PLOTCONFIG: PlotConfig = { main_plot: {}, subplots: {} };
