/** Types mirroring backend/app/schemas/analysis.py */

export interface ParticleResult {
  id: number;
  particle_class: string;
  confidence: number;
  area_pixels: number;
  perimeter_pixels: number;
  width_pixels: number;
  height_pixels: number;
  aspect_ratio: number;
  circularity: number;
  solidity: number;
  extent: number;
  equivalent_diameter_pixels: number;
  mean_R: number;
  mean_G: number;
  mean_B: number;
  equivalent_diameter_micrometers: number | null;
  length_micrometers: number | null;
  width_micrometers: number | null;
}

export interface SizeSummary {
  count: number;
  mean?: number | null;
  min?: number | null;
  max?: number | null;
  median?: number | null;
  std?: number | null;
}

export interface SizeDistribution {
  unit: "pixels" | "micrometers";
  histogram: { bin_edges: number[]; counts: number[] };
  summary: SizeSummary;
}

export interface AnalyzeResponse {
  status: "success" | "success_with_warnings";
  image_width: number;
  image_height: number;
  total_candidates: number;
  suspected_microplastics: number;
  class_counts: Record<string, number>;
  particles: ParticleResult[];
  annotated_image: string | null;
  original_image: string | null;
  processed_image: string | null;
  size_distribution: SizeDistribution | null;
  calibration: { calibrated: boolean; pixels_per_micrometer: number };
  warnings: string[];
  disclaimer: string;
}

export interface ModelStatus {
  model_available: boolean;
  model_name: string;
  classes: string[];
  feature_names: string[] | null;
  message?: string | null;
}

export interface AnalyzeOptions {
  pixelsPerMicrometer: number;
  minParticleAreaPx: number | null;
  thresholdInvert: boolean | null;
}
