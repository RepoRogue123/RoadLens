import { useEffect, useMemo, useState } from 'react';
import { motion as Motion } from 'motion/react';
import { Bike, Car, Truck, CircleDot, Ruler } from 'lucide-react';
import InstrumentPanel from '../instrument/InstrumentPanel';
import { allDrops, rollPath } from '../../utils/wheelDrop';

const TYRE_STYLE = {
  bicycle: { color: '#a3e635', Icon: Bike },
  scooter: { color: '#22d3ee', Icon: CircleDot },
  auto_rickshaw: { color: '#facc15', Icon: CircleDot },
  motorcycle: { color: '#fb923c', Icon: CircleDot },
  car: { color: '#f472b6', Icon: Car },
  suv: { color: '#c084fc', Icon: Car },
  truck: { color: '#94a3b8', Icon: Truck },
};

// Median width of the measured PothRGBD test potholes (scripts/eval_wheel_drop.py): the
// starting point when the photo carries no scale. Always labelled as assumed.
const ASSUMED_WIDTH_MM = 600;
const PASS_MS = 4200;
const PAUSE_MS = 700;

function useLoop(key) {
  const [t, setT] = useState(0);
  useEffect(() => {
    let raf;
    const start = performance.now();
    const tick = (now) => {
      const cycle = (now - start) % (PASS_MS + PAUSE_MS);
      setT(Math.min(1, cycle / PASS_MS));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [key]);
  return t;
}

function RollingWheel({ patch, tyre, axis, mmPerCell, exaggerate }) {
  const path = useMemo(() => rollPath(patch, tyre, axis, mmPerCell), [patch, tyre, axis, mmPerCell]);
  const t = useLoop(`${tyre.id}-${mmPerCell.toFixed(3)}-${axis}`);
  const R = tyre.radius_mm;
  const L = path.profileMm;
  const deepest = Math.max(1, -Math.min(...path.surface));
  const ex = exaggerate;

  // View: the pothole plus a margin, true scale (times `ex` vertically), wheel top clipped.
  const m = Math.max(0.15 * L, 0.35 * R);
  const vx0 = -m;
  const vw = L + 2 * m;
  const bottom = (deepest * 1.3 + 8) * ex;
  const vh = Math.max(vw / 2.6, bottom * 2.2);
  const top = vh - bottom;

  // Wheel position along the padded path (eased), centre height interpolated.
  const ease = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
  const xStart = vx0 - R;
  const xEnd = vx0 + vw + R;
  const xc = xStart + (xEnd - xStart) * ease;
  const step = mmPerCell;
  const fi = Math.min(path.x.length - 1, Math.max(0, (xc - path.x[0]) / step));
  const i0 = Math.floor(fi);
  const i1 = Math.min(path.x.length - 1, i0 + 1);
  const c = path.centre[i0] + (path.centre[i1] - path.centre[i0]) * (fi - i0);
  const dip = Math.max(0, R - c);
  const maxDrop = Math.max(0, R - Math.min(...path.centre));

  const Y = (mm) => -mm * ex;
  const road = path.x
    .map((x, i) => `${x.toFixed(1)},${Y(path.surface[i]).toFixed(2)}`)
    .join(' ');
  const roadPoly = `${path.x[0]},${bottom + 50} ${road} ${path.x[path.x.length - 1]},${bottom + 50}`;
  const trace = path.x
    .map((x, i) => (x <= xc ? `${x.toFixed(1)},${Y(path.centre[i] - R).toFixed(2)}` : null))
    .filter(Boolean)
    .join(' ');
  const angle = xc / R;
  const rim = 0.62 * R;
  const style = TYRE_STYLE[tyre.id] || TYRE_STYLE.car;

  return (
    <div className="relative">
      <svg viewBox={`${vx0} ${-top} ${vw} ${vh}`} className="w-full h-auto block" role="img"
           aria-label={`${tyre.label} rolling over the pothole cross-section`}>
        <defs>
          <linearGradient id="asphalt" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#475569" />
            <stop offset="100%" stopColor="#1e293b" />
          </linearGradient>
          <clipPath id="view">
            <rect x={vx0} y={-top} width={vw} height={vh} />
          </clipPath>
        </defs>
        <g clipPath="url(#view)">
          <polygon points={roadPoly} fill="url(#asphalt)" />
          <polyline points={road} fill="none" stroke="#e2e8f0" strokeWidth={vw / 500} />
          <line x1={vx0} x2={vx0 + vw} y1={Y(-maxDrop)} y2={Y(-maxDrop)}
                stroke={style.color} strokeDasharray={`${vw / 120} ${vw / 160}`} strokeWidth={vw / 700} opacity="0.7" />
          {trace && <polyline points={trace} fill="none" stroke={style.color} strokeWidth={vw / 350} opacity="0.9" />}
          <g>
            <ellipse cx={xc} cy={Y(c)} rx={R} ry={R * ex} fill="#0b1220" stroke="#64748b" strokeWidth={vw / 400} />
            <ellipse cx={xc} cy={Y(c)} rx={R * 0.96} ry={R * 0.96 * ex} fill="none" stroke="#1f2937" strokeWidth={R * 0.07} />
            <ellipse cx={xc} cy={Y(c)} rx={rim} ry={rim * ex} fill="#111827" stroke={style.color} strokeWidth={vw / 450} />
            {[0, 1, 2, 3, 4].map((k) => {
              const th = -angle + (k * 2 * Math.PI) / 5;
              return (
                <line key={k} x1={xc} y1={Y(c)} x2={xc + rim * Math.cos(th)} y2={Y(c) - rim * ex * Math.sin(th)}
                      stroke={style.color} strokeWidth={vw / 500} opacity="0.75" />
              );
            })}
            <circle cx={xc} cy={Y(c)} r={vw / 160} fill={style.color} />
          </g>
        </g>
        <text x={vx0 + vw * 0.012} y={Y(-maxDrop) - vh * 0.02} fill={style.color} fontSize={vh * 0.055} fontFamily="monospace">
          max drop {maxDrop.toFixed(0)} mm
        </text>
      </svg>
      <div className="absolute top-2 right-3 text-right font-mono pointer-events-none">
        <div className="text-[10px] uppercase tracking-[0.14em] text-slate-400">falling now</div>
        <div className="text-2xl font-bold" style={{ color: style.color }}>{dip.toFixed(0)} mm</div>
      </div>
    </div>
  );
}

// The parent keys this panel by pothole, so a new scan starts from fresh state.
export default function WheelDropPanel({ hazard }) {
  const wheel = hazard?.wheel;
  const patch = wheel?.patch;
  const measuredWidth = wheel?.scaled ? wheel.width_mm : null;
  const [selected, setSelected] = useState('car');
  const [widthMm, setWidthMm] = useState(measuredWidth || ASSUMED_WIDTH_MM);
  const [distanceM, setDistanceM] = useState('');
  const [exaggerate, setExaggerate] = useState(1);

  const mmPerCell = patch ? widthMm / patch.equiv_diameter_cells : null;
  const drops = useMemo(
    () => (patch && mmPerCell ? allDrops(patch, wheel.tyres, mmPerCell) : []),
    [patch, wheel, mmPerCell],
  );
  if (!wheel || !patch) return null;

  const tyre = drops.find((d) => d.id === selected) || drops[0];
  const fromDistance = distanceM !== '' && wheel.angular_size;
  const scaleLabel = measuredWidth && widthMm === measuredWidth
    ? 'MEASURED SCALE'
    : fromDistance ? 'SCALE FROM DISTANCE' : 'ASSUMED WIDTH';
  const maxBar = Math.max(1, ...drops.map((d) => d.drop), wheel.deepest_mm);

  const onDistance = (v) => {
    setDistanceM(v);
    const d = parseFloat(v);
    if (wheel.angular_size && d > 0) setWidthMm(Math.round(wheel.angular_size * d * 1000));
  };

  return (
    <InstrumentPanel title="Wheel Drop by Tyre" accent="cyan" statusLabel={scaleLabel} bodyClassName="p-4 sm:p-5" flicker={false}>
      <div className="flex flex-wrap gap-2 mb-4">
        {drops.map((d) => {
          const s = TYRE_STYLE[d.id] || TYRE_STYLE.car;
          const on = d.id === selected;
          return (
            <button key={d.id} type="button" onClick={() => setSelected(d.id)}
                    className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-[2px] border font-mono text-[11px] uppercase tracking-[0.08em] transition-colors ${on ? 'bg-slate-800 text-white' : 'bg-slate-950/50 text-slate-400 hover:text-slate-200'}`}
                    style={{ borderColor: on ? s.color : '#334155' }}>
              <s.Icon className="w-3.5 h-3.5" style={{ color: s.color }} />
              {d.label}
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="lg:col-span-2 bg-slate-950/60 border border-slate-700 rounded-[2px] p-2">
          <RollingWheel patch={patch} tyre={tyre} axis={tyre.axis} mmPerCell={mmPerCell} exaggerate={exaggerate} />
          <div className="flex items-center justify-between px-2 pt-2 font-mono text-[10px] uppercase tracking-[0.12em] text-slate-500">
            <span>{tyre.size} · radius {tyre.radius_mm.toFixed(0)} mm · width {tyre.width_mm.toFixed(0)} mm</span>
            <span className="flex gap-1">
              {[1, 2, 4].map((e) => (
                <button key={e} type="button" onClick={() => setExaggerate(e)}
                        className={`px-1.5 rounded-[2px] border ${exaggerate === e ? 'border-cyan-400 text-cyan-300' : 'border-slate-700'}`}>
                  {e === 1 ? 'true scale' : `depth ×${e}`}
                </button>
              ))}
            </span>
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-slate-950/60 border border-slate-700 rounded-[2px] p-4">
            <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500">{tyre.label} wheel drops</div>
            <div className="text-4xl font-bold font-mono mt-1" style={{ color: (TYRE_STYLE[tyre.id] || {}).color }}>
              {tyre.drop.toFixed(0)}<span className="text-lg text-slate-400"> mm</span>
            </div>
            <p className="text-xs text-slate-400 mt-2 leading-relaxed">
              A wheel cannot follow a hole narrower than itself: it bridges it and falls only as far as its circle fits.
              The pothole is {wheel.deepest_mm.toFixed(0)} mm at its deepest point.
            </p>
          </div>

          <div className="bg-slate-950/60 border border-slate-700 rounded-[2px] p-4 space-y-3">
            <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500">
              <Ruler className="w-3 h-3" /> Pothole width
            </div>
            <input type="range" min="100" max="2000" step="10" value={widthMm}
                   onChange={(e) => { setWidthMm(Number(e.target.value)); setDistanceM(''); }}
                   className="w-full accent-cyan-400" aria-label="Pothole width in millimetres" />
            <div className="flex justify-between font-mono text-sm text-white">
              <span>{widthMm} mm</span>
              {measuredWidth && widthMm !== measuredWidth && (
                <button type="button" className="text-[10px] text-cyan-300 uppercase" onClick={() => setWidthMm(measuredWidth)}>
                  reset to measured
                </button>
              )}
            </div>
            {wheel.angular_size && (
              <label className="block font-mono text-[10px] uppercase tracking-[0.12em] text-slate-500">
                or camera-to-pothole distance (m)
                <input type="number" min="0.2" max="30" step="0.1" value={distanceM} placeholder="e.g. 1.5"
                       onChange={(e) => onDistance(e.target.value)}
                       className="mt-1 w-full bg-slate-900 border border-slate-700 rounded-[2px] px-2 py-1 text-sm text-white normal-case" />
              </label>
            )}
            <p className="text-[11px] text-slate-500 leading-relaxed">
              {scaleLabel === 'MEASURED SCALE' && 'Width from the photo’s angular size and the camera distance you supplied.'}
              {scaleLabel === 'SCALE FROM DISTANCE' && 'Width = angular size from the photo × your distance (within about 10% on measured 3D).'}
              {scaleLabel === 'ASSUMED WIDTH' && `A photo alone has no horizontal scale. Starting at ${ASSUMED_WIDTH_MM} mm (the median measured pothole); whatever the width, no wheel drops more than ${wheel.deepest_mm.toFixed(0)} mm.`}
            </p>
          </div>
        </div>
      </div>

      <div className="mt-5 bg-slate-950/60 border border-slate-700 rounded-[2px] p-4">
        <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500 mb-3">Every tyre, same pothole</div>
        <div className="space-y-2">
          {drops.map((d) => {
            const s = TYRE_STYLE[d.id] || TYRE_STYLE.car;
            return (
              <button key={d.id} type="button" onClick={() => setSelected(d.id)} className="w-full flex items-center gap-3 group">
                <span className={`w-36 shrink-0 text-left font-mono text-[11px] ${d.id === selected ? 'text-white' : 'text-slate-400 group-hover:text-slate-200'}`}>
                  {d.label}
                </span>
                <span className="relative flex-1 h-4 bg-slate-900 rounded-[1px] overflow-hidden">
                  <Motion.span className="absolute inset-y-0 left-0" style={{ background: s.color, opacity: d.id === selected ? 1 : 0.55 }}
                               initial={false} animate={{ width: `${(100 * d.drop) / maxBar}%` }}
                               transition={{ type: 'spring', stiffness: 120, damping: 20 }} />
                </span>
                <span className="w-16 text-right font-mono text-xs text-slate-300">{d.drop.toFixed(0)} mm</span>
              </button>
            );
          })}
        </div>
        {wheel.validation && (
          <p className="mt-3 text-[11px] text-slate-500 leading-relaxed">
            Checked against measured 3D on {wheel.validation.n} held-out potholes: a car wheel&rsquo;s predicted drop
            was within {wheel.validation.car_mae_mm.toFixed(0)} mm of the measured one on average (correlation{' '}
            {wheel.validation.car_r.toFixed(2)}).
          </p>
        )}
      </div>
    </InstrumentPanel>
  );
}
