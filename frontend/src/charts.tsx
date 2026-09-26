import { useEffect, useMemo, useRef } from "react";
import * as echarts from "echarts";
import type { AnalyzeResponse } from "./types";

/** Bar chart of the particle-size histogram returned by the backend. */
export function SizeHistogram({ result }: { result: AnalyzeResponse }) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);

  const dist = result.size_distribution;

  const option = useMemo<echarts.EChartsOption>(() => {
    if (!dist || dist.histogram.bin_edges.length === 0) {
      return { title: { text: "No data", left: "center", textStyle: { color: "#6c8b7d" } } };
    }
    const edges = dist.histogram.bin_edges;
    const labels = edges.slice(0, -1).map((edge, i) => {
      const lo = edge.toFixed(1);
      const hi = edges[i + 1].toFixed(1);
      return `${lo}–${hi}`;
    });
    return {
      grid: { left: 42, right: 16, top: 30, bottom: 46 },
      tooltip: { trigger: "axis" },
      xAxis: {
        type: "category",
        data: labels,
        name: dist.unit === "micrometers" ? "µm" : "px",
        nameTextStyle: { color: "#6c8b7d" },
        axisLabel: { color: "#35594a", rotate: 35, fontSize: 10 },
        axisLine: { lineStyle: { color: "#a8ecc9" } },
      },
      yAxis: {
        type: "value",
        name: "particles",
        nameTextStyle: { color: "#6c8b7d" },
        axisLabel: { color: "#35594a" },
        splitLine: { lineStyle: { color: "rgba(168,236,201,0.4)" } },
      },
      series: [
        {
          type: "bar",
          data: dist.histogram.counts,
          barCategoryGap: "25%",
          itemStyle: {
            borderRadius: [6, 6, 0, 0],
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: "#27a56b" },
              { offset: 1, color: "#0f5132" },
            ]),
          },
        },
      ],
    };
  }, [dist]);

  useEffect(() => {
    if (!ref.current) return;
    chart.current = echarts.init(ref.current);
    chart.current.setOption(option);
    const onResize = () => chart.current?.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.current?.dispose();
      chart.current = null;
    };
  }, [option]);

  return <div ref={ref} style={{ width: "100%", height: 280 }} aria-label="Size histogram" />;
}

/** Donut chart of class counts. */
export function ClassDonut({ result }: { result: AnalyzeResponse }) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);

  const option = useMemo<echarts.EChartsOption>(() => {
    const palette: Record<string, string> = {
      fiber: "#27a56b",
      fragment: "#0f5132",
      bead: "#63d99a",
      other: "#c9d8d0",
      unclassified: "#c9d8d0",
    };
    const data = Object.entries(result.class_counts)
      .filter(([, value]) => value > 0)
      .map(([name, value]) => ({
        name,
        value,
        itemStyle: { color: palette[name] ?? "#a8ecc9" },
      }));
    return {
      tooltip: { trigger: "item" },
      legend: { bottom: 0, textStyle: { color: "#35594a", fontSize: 11 } },
      series: [
        {
          type: "pie",
          radius: ["48%", "72%"],
          center: ["50%", "44%"],
          label: { show: false },
          data: data.length > 0 ? data : [{ name: "none", value: 1, itemStyle: { color: "#e5efe9" } }],
        },
      ],
    };
  }, [result.class_counts]);

  useEffect(() => {
    if (!ref.current) return;
    chart.current = echarts.init(ref.current);
    chart.current.setOption(option);
    const onResize = () => chart.current?.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      chart.current?.dispose();
      chart.current = null;
    };
  }, [option]);

  return <div ref={ref} style={{ width: "100%", height: 280 }} aria-label="Class distribution" />;
}
