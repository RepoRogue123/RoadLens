import { BrainCircuit, Mountain } from 'lucide-react';

/** Normal deviation is reported in degrees; this is the gauge's full-scale value. */
const NORMAL_DEVIATION_FULL_SCALE_DEG = 45;
import InstrumentPanel from '../instrument/InstrumentPanel';

/** Semicircular arc gauge with a mono value — instrument dial. */
function ArcGauge({ label, value, max = 1.0, color = '#a78bfa', format = (v) => v.toFixed(4) }) {
  const pct = Math.max(0, Math.min(value / max, 1));
  const R = 34;
  const C = Math.PI * R; // semicircle length
  return (
    <div className="flex items-center gap-4">
      <svg width="88" height="52" viewBox="0 0 88 52" aria-hidden="true" className="shrink-0">
        <path
          d={`M 10 46 A ${R} ${R} 0 0 1 78 46`}
          fill="none"
          stroke="#29303f"
          strokeWidth="6"
          strokeLinecap="round"
        />
        <path
          d={`M 10 46 A ${R} ${R} 0 0 1 78 46`}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={`${C * pct} ${C}`}
          style={{ filter: `drop-shadow(0 0 4px ${color}66)`, transition: 'stroke-dasharray 0.8s ease-out' }}
        />
        {/* Quadrant ticks */}
        <line x1="44" y1="6" x2="44" y2="12" stroke="#3c4354" strokeWidth="1.5" />
        <line x1="10" y1="46" x2="14" y2="46" stroke="#3c4354" strokeWidth="1.5" />
        <line x1="74" y1="46" x2="78" y2="46" stroke="#3c4354" strokeWidth="1.5" />
      </svg>
      <div className="min-w-0">
        <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500">{label}</p>
        <p className="font-mono text-lg font-semibold" style={{ color }}>
          {format(value)}
        </p>
      </div>
    </div>
  );
}

export default function SemanticIntelligence({ geometryAnalysis }) {
  if (!geometryAnalysis) return null;

  const { foundationFeatures, curvatureFeatures } = geometryAnalysis;
  // Mean angle (degrees) between surface normals on the boundary ring and the
  // road's reference normal, from features.extract_surface_normal_features.
  // Normals are Sobel derivatives of the monocular depth map — this is NOT
  // shape-from-shading, and it inherits the depth model's failures.
  const normalDeviationDeg = curvatureFeatures?.mean_normal_deviation || 0;

  return (
    <InstrumentPanel
      title="Semantic Verification"
      accent="holo"
      statusLabel="DINOV2 · NORMALS"
      bodyClassName="p-4 sm:p-5"
      flicker={false}
    >
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Foundation Features (DINOv2) */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <div className="p-1.5 rounded-[2px] bg-holo-violet/10 border border-holo-violet/30">
              <BrainCircuit className="w-4 h-4 text-holo-violet-bright" />
            </div>
            <h3 className="text-sm font-semibold text-slate-300">Vision foundation — is it a real crater?</h3>
          </div>

          <div className="bg-slate-950/60 rounded-[2px] p-4 border border-slate-700 min-h-[150px] space-y-4">
            {foundationFeatures ? (
              <>
                <ArcGauge
                  label="Semantic dissimilarity"
                  value={foundationFeatures.dissimilarity}
                  max={1.0}
                  color="#a78bfa"
                />
                <ArcGauge
                  label="Interior variance"
                  value={foundationFeatures.insideVariance}
                  max={2.0}
                  color="#e879f9"
                />
              </>
            ) : (
              <div className="h-full min-h-[120px] flex items-center justify-center font-mono text-[11px] text-slate-500 text-center uppercase tracking-wider">
                Foundation model offline
              </div>
            )}
          </div>
        </div>

        {/* Boundary surface normals */}
        <div>
          <div className="flex items-center gap-2 mb-4">
            <div className="p-1.5 rounded-[2px] bg-holo-teal/10 border border-holo-teal/30">
              <Mountain className="w-4 h-4 text-holo-teal-bright" />
            </div>
            <h3 className="text-sm font-semibold text-slate-300">Surface normals — wall steepness</h3>
          </div>

          <div className="bg-slate-950/60 rounded-[2px] p-4 border border-slate-700 min-h-[150px]">
            <ArcGauge
              label="Boundary normal deviation"
              value={normalDeviationDeg}
              max={NORMAL_DEVIATION_FULL_SCALE_DEG}
              color="#5eead4"
              format={(v) => `${v.toFixed(1)}°`}
            />
            <p className="mt-4 text-xs text-slate-500 leading-relaxed">
              Mean tilt of the surface around the rim relative to the road, derived from the
              monocular depth map. Steeper walls read higher; a flat stain reads near zero only
              if the depth model also sees it as flat.
            </p>
          </div>
        </div>
      </div>
    </InstrumentPanel>
  );
}
