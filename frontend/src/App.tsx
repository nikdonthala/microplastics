import { useCallback, useMemo, useRef, useState } from "react";
import {
  ApiError,
  analyzeImage,
  fetchHealth,
  fetchModelMetrics,
  fetchModelStatus,
} from "./api";
import { ClassDonut, SizeHistogram } from "./charts";
import type { AnalyzeResponse, ModelMetrics, ModelStatus } from "./types";

type Phase = "idle" | "working" | "done" | "error";

const CLASS_ICON: Record<string, string> = {
  fiber: "🧵",
  fragment: "🔷",
  bead: "⚪",
  other: "◾",
  unclassified: "❔",
};

export default function App() {
  const [health, setHealth] = useState<string | null>(null);
  const [model, setModel] = useState<ModelStatus | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  // Analysis options
  const [calibration, setCalibration] = useState("");
  const [minArea, setMinArea] = useState("");
  const [invert, setInvert] = useState<"auto" | "dark-on-light" | "light-on-dark">("auto");

  const inputRef = useRef<HTMLInputElement>(null);

  const loadStatuses = useCallback(async () => {
    try {
      const [h, m, mm] = await Promise.all([
        fetchHealth(),
        fetchModelStatus(),
        fetchModelMetrics(),
      ]);
      setHealth(h.status === "ok" ? `API v${h.version}` : "degraded");
      setModel(m);
      setMetrics(m.model_available ? mm : null);
    } catch {
      setHealth("offline");
    }
  }, []);

  // statuses are fetched once on mount and re-checked after each analysis
  const [statusesLoaded, setStatusesLoaded] = useState(false);
  if (!statusesLoaded) {
    setStatusesLoaded(true);
    void loadStatuses();
  }

  const acceptFile = useCallback((candidate: File | null | undefined) => {
    if (!candidate) return;
    if (!/\.(jpe?g|png)$/i.test(candidate.name)) {
      setError("Unsupported file type. Please upload a JPG, JPEG or PNG image.");
      return;
    }
    setError(null);
    setResult(null);
    setPhase("idle");
    setFile(candidate);
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDrag(false);
      acceptFile(event.dataTransfer.files?.[0]);
    },
    [acceptFile],
  );

  const analyze = useCallback(async () => {
    if (!file) return;
    setPhase("working");
    setError(null);
    try {
      const response = await analyzeImage(file, {
        pixelsPerMicrometer: Number(calibration) > 0 ? Number(calibration) : 0,
        minParticleAreaPx: minArea !== "" ? Number(minArea) : null,
        thresholdInvert:
          invert === "dark-on-light" ? true : invert === "light-on-dark" ? false : null,
      });
      setResult(response);
      setPhase("done");
      void loadStatuses();
    } catch (exc) {
      setError(exc instanceof ApiError ? exc.message : "Analysis failed. Is the backend running?");
      setPhase("error");
    }
  }, [file, calibration, minArea, invert, loadStatuses]);

  const stats = useMemo(() => {
    if (!result) return null;
    const dist = result.size_distribution;
    return [
      { label: "Candidate particles", value: String(result.total_candidates) },
      { label: "Suspected microplastics", value: String(result.suspected_microplastics) },
      {
        label: `Mean diameter (${dist?.unit ?? "px"})`,
        value: dist?.summary?.mean != null ? String(dist.summary.mean) : "—",
      },
      {
        label: `Median (${dist?.unit ?? "px"})`,
        value: dist?.summary?.median != null ? String(dist.summary.median) : "—",
      },
    ];
  }, [result]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark">◉</div>
          <div>
            <h1>MicroScan AI</h1>
            <p>Image-based screening of suspected microplastics in water samples</p>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          {model?.model_available && (
            <div className="badge muted">{`model: ${model.model_name}`}</div>
          )}
          <div className="badge">
            <span className="dot" />
            {health ?? "connecting…"}
          </div>
        </div>
      </header>

      {/* ------------------------------------------------------ upload */}
      <section className="glass panel">
        <h2 className="panel-title">Analyze a micrograph</h2>
        <p className="panel-sub">
          Upload a bright-field microscope image (JPG/PNG). Processing happens server-side;
          nothing is stored.
        </p>

        <div
          className={`dropzone${drag ? " drag" : ""}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
          }}
        >
          <div className="icon">🧫</div>
          {file ? (
            <>
              <h3>{file.name}</h3>
              <p>
                {(file.size / 1024 / 1024).toFixed(2)} MB — click to choose a different image
              </p>
            </>
          ) : (
            <>
              <h3>Drop an image here</h3>
              <p>or click to browse — JPG, JPEG or PNG</p>
            </>
          )}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png"
          hidden
          onChange={(e) => acceptFile(e.target.files?.[0])}
        />

        <div className="controls-grid">
          <div className="field">
            <label htmlFor="cal">Calibration (px / µm) — optional</label>
            <input
              id="cal"
              className="input"
              inputMode="decimal"
              placeholder="e.g. 0.5"
              value={calibration}
              onChange={(e) => setCalibration(e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="minarea">Min particle area (px) — optional</label>
            <input
              id="minarea"
              className="input"
              inputMode="numeric"
              placeholder="default 30"
              value={minArea}
              onChange={(e) => setMinArea(e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="invert">Particles in image</label>
            <select
              id="invert"
              className="select"
              value={invert}
              onChange={(e) => setInvert(e.target.value as typeof invert)}
            >
              <option value="auto">Auto (server default)</option>
              <option value="dark-on-light">Dark particles on light background</option>
              <option value="light-on-dark">Light particles on dark background</option>
            </select>
          </div>
          <div className="field">
            <label>&nbsp;</label>
            <button className="btn btn-primary" onClick={analyze} disabled={!file || phase === "working"}>
              {phase === "working" ? (
                <>
                  <span className="spinner" /> Analyzing…
                </>
              ) : (
                <>Run analysis</>
              )}
            </button>
          </div>
        </div>

        {metrics && <ModelCard metrics={metrics} />}
        {model && !model.model_available && model.message && (
          <div className="callout info" style={{ marginTop: 14 }}>
            {model.message}
          </div>
        )}
        {error && (
          <div className="callout" style={{ marginTop: 14 }}>
            {error}
          </div>
        )}
      </section>

      {/* ------------------------------------------------------ results */}
      {result && stats && (
        <>
          <section className="glass panel">
            <h2 className="panel-title">Results</h2>
            <p className="panel-sub">
              {result.image_width}×{result.image_height} px ·{" "}
              {result.calibration.calibrated
                ? `calibrated at ${result.calibration.pixels_per_micrometer} px/µm`
                : "uncalibrated (pixel units)"}
            </p>
            <div className="stat-grid">
              {stats.map((s) => (
                <div key={s.label} className="glass stat-card">
                  <div className="value">{s.value}</div>
                  <div className="label">{s.label}</div>
                </div>
              ))}
            </div>
            {result.warnings.length > 0 && (
              <div className="callout" style={{ marginTop: 16 }}>
                <strong>Notes &amp; limitations</strong>
                <ul>
                  {result.warnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              </div>
            )}
            <p className="disclaimer">{result.disclaimer}</p>
          </section>

          <section className="glass panel">
            <h2 className="panel-title">Images</h2>
            <p className="panel-sub">Left to right: annotated, original, processed mask.</p>
            <div className="image-grid">
              {(
                [
                  ["Annotated", result.annotated_image],
                  ["Original", result.original_image],
                  ["Processed mask", result.processed_image],
                ] as const
              ).map(([title, src]) =>
                src ? (
                  <figure key={title} className="image-card" style={{ margin: 0 }}>
                    <h4>{title}</h4>
                    <img src={src} alt={title} />
                  </figure>
                ) : null,
              )}
            </div>
          </section>

          <section className="glass panel">
            <h2 className="panel-title">Distributions</h2>
            <p className="panel-sub">Morphology classes and particle-size spread.</p>
            <div className="charts-grid">
              <div className="chart-box">
                <h4>Size histogram ({result.size_distribution?.unit ?? "pixels"})</h4>
                <SizeHistogram result={result} />
              </div>
              <div className="chart-box">
                <h4>Class distribution</h4>
                <ClassDonut result={result} />
              </div>
            </div>
          </section>

          <section className="glass panel">
            <h2 className="panel-title">Detected particles</h2>
            <p className="panel-sub">
              Confidence is the Random Forest vote share — not a calibrated probability.
            </p>
            {result.particles.length === 0 ? (
              <p className="muted center">No particles passed the detection filters.</p>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.84rem" }}>
                  <thead>
                    <tr>
                      {["#", "Class", "Conf.", "Area (px²)", "Ø eq. (px)", "Ø eq. (µm)", "Circularity", "Aspect ratio"].map(
                        (h) => (
                          <th
                            key={h}
                            style={{
                              textAlign: "left",
                              padding: "8px 10px",
                              borderBottom: "1px solid rgba(25,135,84,0.2)",
                              color: "var(--ink-soft)",
                              fontWeight: 600,
                              whiteSpace: "nowrap",
                            }}
                          >
                            {h}
                          </th>
                        ),
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {result.particles.map((p) => (
                      <tr key={p.id}>
                        <td style={cellStyle}>#{p.id}</td>
                        <td style={cellStyle}>
                          {CLASS_ICON[p.particle_class] ?? "•"} {p.particle_class}
                        </td>
                        <td style={cellStyle}>{(p.confidence * 100).toFixed(0)}%</td>
                        <td style={cellStyle}>{p.area_pixels.toLocaleString()}</td>
                        <td style={cellStyle}>{p.equivalent_diameter_pixels}</td>
                        <td style={cellStyle}>{p.equivalent_diameter_micrometers ?? "—"}</td>
                        <td style={cellStyle}>{p.circularity}</td>
                        <td style={cellStyle}>{p.aspect_ratio}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}

      <footer className="disclaimer center site-footer">
        MicroScan AI · educational prototype · image analysis alone does not confirm polymer
        identity (FTIR/Raman required)
        <br />
        <a
          href="https://github.com/nikdonthala/microplastics"
          target="_blank"
          rel="noopener noreferrer"
        >
          Github repo
        </a>
        {" · "}
        Created by{" "}
        <a href="https://github.com/nikdonthala" target="_blank" rel="noopener noreferrer">
          nikdonthala
        </a>
      </footer>
    </div>
  );
}

const cellStyle: React.CSSProperties = {
  padding: "8px 10px",
  borderBottom: "1px solid rgba(25,135,84,0.10)",
  whiteSpace: "nowrap",
};

/** Model provenance card: algorithm, test metrics and CV selection table. */
function ModelCard({ metrics }: { metrics: ModelMetrics }) {
  const candidates = (metrics.candidates ?? []).slice().sort(
    (a, b) => b.cv_mean_f1_macro - a.cv_mean_f1_macro,
  );
  const importances = Object.entries(metrics.feature_importances ?? {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);
  const maxImportance = importances[0]?.[1] ?? 1;
  return (
    <div className="callout info" style={{ marginTop: 14 }}>
      <strong>Trained model — {metrics.algorithm ?? "classifier"}</strong>
      <span style={{ opacity: 0.85 }}>
        {" "}
        · test accuracy {(100 * (metrics.accuracy ?? 0)).toFixed(1)}% · macro-F1{" "}
        {(metrics.f1_macro ?? 0).toFixed(3)} · {metrics.n_training_samples ?? "?"} training /{" "}
        {metrics.n_test_samples ?? "?"} test samples · dataset {metrics.dataset ?? "?"}
      </span>
      {candidates.length > 0 && (
        <ul>
          {candidates.map((row) => (
            <li key={row.algorithm}>
              {row.algorithm}: CV macro-F1 {row.cv_mean_f1_macro.toFixed(4)} ±{" "}
              {row.cv_std_f1_macro.toFixed(4)} ({row.cv_folds} folds)
            </li>
          ))}
        </ul>
      )}
      {importances.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: "0.74rem", fontWeight: 600, marginBottom: 4 }}>
            Top features (model importance)
          </div>
          {importances.map(([name, value]) => (
            <div key={name} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
              <span style={{ width: 150, fontSize: "0.74rem" }}>{name}</span>
              <div
                style={{
                  height: 6,
                  flex: 1,
                  borderRadius: 999,
                  background: "rgba(25,135,84,0.12)",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${(value / maxImportance) * 100}%`,
                    borderRadius: 999,
                    background: "linear-gradient(90deg, var(--green-400), var(--green-600))",
                  }}
                />
              </div>
              <span style={{ width: 46, fontSize: "0.72rem", textAlign: "right" }}>
                {value.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
