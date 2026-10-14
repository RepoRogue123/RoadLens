import { useMemo, useState } from 'react';
import { motion as Motion } from 'motion/react';
import { X, ArrowRight, Droplets, Layers, Crosshair, Gauge, Target, AlertTriangle } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import InstrumentPanel from '../instrument/InstrumentPanel';
import CountUp from './CountUp';
import { SEVERITY } from '../../theme/severity';
import { panelEnter } from '../../theme/motion';

const BANDS = ['Shallow', 'Moderate', 'Deep'];
const MONO = 'IBM Plex Mono, monospace';
const TICK = { fill: '#97a0b1', fontSize: 10, fontFamily: MONO };
const TOOLTIP_STYLE = {
  background: 'rgba(10,12,16,0.95)',
  border: '1px solid #3c4354',
  borderRadius: 2,
  fontSize: 11,
  fontFamily: MONO,
  color: '#cdd3dd',
};
const KIND_COLOR = {
  existing: '#64748b',
  ablation: '#8b5cf6',
  floor: '#3c4354',
  'ours-earlier': '#22d3ee',
  ours: '#f59e0b',
};
const KIND_LABEL = {
  existing: 'Existing model, used as-is',
  ablation: 'Same network, trained on absolute depth',
  floor: 'No model: always guess the median',
  'ours-earlier': 'Our earlier versions',
  ours: 'Served today',
};

/** Panels slide in as they scroll into view. */
function Reveal({ children, className = '', delay = 0 }) {
  return (
    <Motion.div
      className={className}
      variants={panelEnter}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: '-60px' }}
      transition={{ delay }}
    >
      {children}
    </Motion.div>
  );
}

function Chip({ active, onClick, children, tone = 'amber' }) {
  const on =
    tone === 'cyan'
      ? 'border-cyan-500/40 text-cyan-300 bg-cyan-500/10'
      : 'border-amber-500/40 text-amber-300 bg-amber-500/10';
  return (
    <button
      type="button"
      onClick={onClick}
      className={`px-2.5 py-1 rounded-[2px] border font-mono text-[10px] uppercase tracking-[0.1em] transition-colors ${
        active ? on : 'border-slate-700 text-slate-500 hover:text-slate-300'
      }`}
    >
      {children}
    </button>
  );
}

function Explain({ children }) {
  return <p className="text-xs text-slate-400 leading-relaxed mt-3">{children}</p>;
}

/* ── 1. Headline ─────────────────────────────────────────────────────────── */

function Headline({ h, table }) {
  const old = table?.find((r) => r.name.startsWith('Old'));
  const ours = table?.find((r) => r.name.startsWith('Depth network'));
  const tiles = [
    { icon: Gauge, label: 'Mean depth error', value: h.maeMm, dec: 1, unit: 'mm', color: '#f59e0b',
      note: `${h.nTest} potholes from capture sessions never seen in training` },
    { icon: Target, label: 'Within ±10 mm', value: (h.within10 ?? 0) * 100, dec: 0, unit: '%', color: '#22d3ee',
      note: `80% of potholes within ±${h.q80Mm} mm` },
    { icon: Crosshair, label: 'Correlation with measured', value: h.r, dec: 2, unit: 'r', color: '#a78bfa',
      note: 'bigger measured depth → bigger prediction' },
    { icon: AlertTriangle, label: 'Deep potholes caught', value: h.deepFound, dec: 0, unit: `/ ${h.nDeep}`, color: '#ef4444',
      note: `Moderate from ${h.moderateFromMm} mm · Deep from ${h.deepFromMm} mm` },
  ];
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {tiles.map((t, i) => (
          <Reveal key={t.label} delay={i * 0.06}>
            <InstrumentPanel accent={i === 3 ? 'red' : i === 2 ? 'holo' : i === 1 ? 'cyan' : 'amber'} flicker={false} bodyClassName="p-4">
              <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500">
                <t.icon className="w-3.5 h-3.5" style={{ color: t.color }} /> {t.label}
              </div>
              <p className="font-mono font-semibold text-3xl mt-2 leading-none" style={{ color: t.color }}>
                <CountUp value={t.value} decimals={t.dec} />
                <span className="text-sm text-slate-500 ml-1.5">{t.unit}</span>
              </p>
              <p className="text-[11px] text-slate-500 mt-2 leading-snug">{t.note}</p>
            </InstrumentPanel>
          </Reveal>
        ))}
      </div>

      {old && ours && (
        <Reveal>
          <InstrumentPanel accent="cyan" flicker={false} bodyClassName="p-4">
            <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500 mb-3">
              Severity verdict right, against measured depth
            </p>
            {[old, ours].map((row, i) => (
              <div key={row.name} className="flex items-center gap-3 mb-2 last:mb-0">
                <span className={`w-48 shrink-0 text-xs ${i ? 'text-amber-300' : 'text-slate-400'}`}>
                  {i ? 'Depth network (today)' : 'Old pipeline'}
                </span>
                <div className="flex-1 h-5 bg-slate-950 border border-slate-800 rounded-[2px] overflow-hidden">
                  <Motion.div
                    className="h-full"
                    style={{ background: i ? 'linear-gradient(90deg,#f59e0b,#fbbf24)' : '#475569' }}
                    initial={{ width: 0 }}
                    whileInView={{ width: `${row.correct * 100}%` }}
                    viewport={{ once: true }}
                    transition={{ duration: 1.1, delay: 0.2 + i * 0.25, ease: [0.22, 1, 0.36, 1] }}
                  />
                </div>
                <span className="w-28 shrink-0 font-mono text-xs text-right text-slate-300">
                  <CountUp value={row.correct * 100} decimals={1} suffix="%" /> · n={row.n}
                </span>
              </div>
            ))}
            <Explain>
              The old pipeline (five classifiers voting on Depth-Anything features) scored 99% against labels it made
              itself. Scored against real measured depth, its verdict was right {(old.correct * 100).toFixed(1)}% of the
              time — worse than always answering the most common band.
            </Explain>
          </InstrumentPanel>
        </Reveal>
      )}
    </div>
  );
}

/* ── 2. Predicted vs measured ────────────────────────────────────────────── */

function ScatterTip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  const err = p.pred - p.gt;
  return (
    <div style={TOOLTIP_STYLE} className="px-3 py-2">
      <p>
        measured <span style={{ color: SEVERITY[p.band].color }}>{p.gt} mm ({p.band})</span>
      </p>
      <p>predicted <span className="text-amber-300">{p.pred} mm</span></p>
      <p className={Math.abs(err) <= 10 ? 'text-teal-300' : 'text-red-400'}>
        error {err > 0 ? '+' : ''}
        {err.toFixed(1)} mm
      </p>
    </div>
  );
}

function PredictedVsMeasured({ points, h }) {
  const [mask, setMask] = useState('human');
  const data = useMemo(() => points.filter((p) => p.mask === mask), [points, mask]);
  const top = Math.ceil(Math.max(60, ...points.map((p) => Math.max(p.gt, p.pred))) / 10) * 10;
  const within = data.filter((p) => Math.abs(p.pred - p.gt) <= 10).length;
  return (
    <InstrumentPanel
      title="Predicted vs measured depth · every test pothole"
      accent="amber"
      flicker={false}
      headerRight={
        <div className="flex gap-1.5">
          <Chip active={mask === 'human'} onClick={() => setMask('human')}>Human outline</Chip>
          <Chip active={mask === 'segmenter'} onClick={() => setMask('segmenter')} tone="cyan">Segmenter outline</Chip>
        </div>
      }
      bodyClassName="p-4"
    >
      <ResponsiveContainer width="100%" height={380}>
        <ScatterChart margin={{ top: 10, right: 20, bottom: 28, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#29303f" />
          <XAxis type="number" dataKey="gt" domain={[0, top]} tick={TICK} tickLine={false} axisLine={false}
            label={{ value: 'MEASURED DEPTH (MM, REALSENSE)', fill: '#626b7d', fontSize: 9, fontFamily: MONO, position: 'insideBottom', offset: -14 }} />
          <YAxis type="number" dataKey="pred" domain={[0, top]} tick={TICK} tickLine={false} axisLine={false}
            label={{ value: 'PREDICTED (MM)', fill: '#626b7d', fontSize: 9, fontFamily: MONO, angle: -90, position: 'insideLeft' }} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: top, y: top }]} stroke="#f59e0b" strokeOpacity={0.7} ifOverflow="hidden" />
          <ReferenceLine segment={[{ x: 0, y: 10 }, { x: top - 10, y: top }]} stroke="#2dd4bf" strokeDasharray="4 4" strokeOpacity={0.5} ifOverflow="hidden" />
          <ReferenceLine segment={[{ x: 10, y: 0 }, { x: top, y: top - 10 }]} stroke="#2dd4bf" strokeDasharray="4 4" strokeOpacity={0.5} ifOverflow="hidden" />
          {[h.moderateFromMm, h.deepFromMm].map((c) => (
            <ReferenceLine key={`x${c}`} x={c} stroke="#3c4354" strokeDasharray="2 4" />
          ))}
          {[h.moderateFromMm, h.deepFromMm].map((c) => (
            <ReferenceLine key={`y${c}`} y={c} stroke="#3c4354" strokeDasharray="2 4" />
          ))}
          <Tooltip content={<ScatterTip />} cursor={{ strokeDasharray: '4 4' }} />
          <Legend verticalAlign="top" height={26} wrapperStyle={{ color: '#97a0b1', fontSize: 11, fontFamily: MONO }} />
          {BANDS.map((b) => (
            <Scatter key={b} name={`Measured ${b}`} data={data.filter((p) => p.band === b)} fill={SEVERITY[b].color}
              fillOpacity={0.8} animationDuration={900} />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
      <Explain>
        Each dot is one pothole from a capture session the network never trained on. On the amber line the prediction
        equals the RealSense measurement; between the dashed teal lines it is within ±10 mm ({within} of {data.length}{' '}
        here). Grey dotted lines are the severity cut-offs, so a dot in the box where its measured and predicted bands
        agree gets the right verdict.
      </Explain>
    </InstrumentPanel>
  );
}

/* ── 3. Against existing approaches ──────────────────────────────────────── */

function Leaderboard({ rows }) {
  const data = rows.map((r) => ({ ...r, short: r.name.replace(', same backbone', '') }));
  return (
    <InstrumentPanel title="Against existing approaches · same 208 potholes" accent="cyan" flicker={false} bodyClassName="p-4">
      <ResponsiveContainer width="100%" height={330}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 56, left: 8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#29303f" horizontal={false} />
          <XAxis type="number" scale="log" domain={[5, 200]} allowDataOverflow ticks={[5, 10, 20, 50, 100, 200]}
            tick={TICK} tickLine={false} axisLine={false} tickFormatter={(v) => `${v}`} />
          <YAxis type="category" dataKey="short" width={190} tick={{ ...TICK, fontSize: 10 }} tickLine={false} axisLine={false} />
          <Tooltip
            cursor={{ fill: 'rgba(255,255,255,0.03)' }}
            contentStyle={TOOLTIP_STYLE}
            formatter={(v, _n, p) => [`${v} mm${p.payload.r != null ? ` · r ${p.payload.r}` : ''}`, KIND_LABEL[p.payload.kind]]}
          />
          <Bar dataKey="maeMm" radius={[0, 2, 2, 0]} animationDuration={1100}>
            {data.map((r) => (
              <Cell key={r.name} fill={KIND_COLOR[r.kind]} />
            ))}
            <LabelList dataKey="maeMm" position="right" formatter={(v) => `${v.toFixed(1)} mm`} style={{ fill: '#cdd3dd', fontSize: 10, fontFamily: MONO }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-1">
        {Object.entries(KIND_LABEL).map(([k, label]) => (
          <span key={k} className="flex items-center gap-1.5 font-mono text-[10px] text-slate-500">
            <span className="w-2.5 h-2.5 rounded-[1px]" style={{ background: KIND_COLOR[k] }} /> {label}
          </span>
        ))}
      </div>
      <Explain>
        Mean error in millimetres (log scale; shorter is better). General depth models used as they come miss by 40–124
        mm because they do not know the road surface or the camera height. Training the same network on absolute depth
        instead of depth below the road plane gives 17 mm — the target choice, not the backbone, does most of the work.
      </Explain>
    </InstrumentPanel>
  );
}

/* ── 4. Which signal follows real depth ──────────────────────────────────── */

/** Bar-end label that sits outside the bar on whichever side it points to. */
function SignedLabel({ x, y, width, height, value }) {
  const left = Math.min(x, x + width);
  const right = Math.max(x, x + width);
  const neg = value < 0;
  return (
    <text x={neg ? left - 6 : right + 6} y={y + height / 2} dy={3.5} textAnchor={neg ? 'end' : 'start'}
      fill={neg ? '#fca5a5' : '#cdd3dd'} fontSize={10} fontFamily={MONO}>
      {value.toFixed(2)}
    </text>
  );
}

function Signals({ rows }) {
  const data = [...rows].sort((a, b) => a.rho - b.rho);
  return (
    <InstrumentPanel title="Which signal actually follows pothole depth?" accent="holo" flicker={false} bodyClassName="p-4">
      <ResponsiveContainer width="100%" height={330}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 44, left: 8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#29303f" horizontal={false} />
          <XAxis type="number" domain={[-0.4, 0.8]} ticks={[-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]} tick={TICK} tickLine={false} axisLine={false} />
          <YAxis type="category" dataKey="name" width={200} tick={{ ...TICK, fontSize: 10 }} tickLine={false} axisLine={false} />
          <ReferenceLine x={0} stroke="#626b7d" />
          <Tooltip cursor={{ fill: 'rgba(255,255,255,0.03)' }} contentStyle={TOOLTIP_STYLE} formatter={(v) => [v, 'rank correlation']} />
          <Bar dataKey="rho" radius={2} animationDuration={1100}>
            {data.map((r) => (
              <Cell key={r.name} fill={r.rho < 0 ? '#ef4444' : r.name.includes('current') ? '#f59e0b' : r.rho > 0.45 ? '#22d3ee' : '#64748b'} />
            ))}
            <LabelList dataKey="rho" content={<SignedLabel />} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <Explain>
        Rank correlation with measured depth over 207 held-out potholes (1 = orders potholes exactly by depth, 0 = no
        relation). The three signals the first pipeline was built on — relative depth, shape from shading and rim
        curvature — sit near zero or below it; pothole size alone does better than any of them.
      </Explain>
    </InstrumentPanel>
  );
}

/* ── 5. Severity: old vs new ─────────────────────────────────────────────── */

function ConfusionGrid({ title, subtitle, block, asPct, accent }) {
  const { matrix } = block;
  return (
    <InstrumentPanel title={title} accent={accent} flicker={false} bodyClassName="p-4">
      <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-slate-500 mb-3">{subtitle}</p>
      <div className="grid grid-cols-[72px_repeat(3,1fr)] gap-1">
        <span />
        {BANDS.map((b) => (
          <span key={b} className="font-mono text-[9px] uppercase tracking-[0.1em] text-center pb-1" style={{ color: SEVERITY[b].color }}>
            said {b}
          </span>
        ))}
        {matrix.map((row, i) => {
          const total = row.reduce((a, b) => a + b, 0) || 1;
          return [
            <span key={`l${i}`} className="font-mono text-[9px] uppercase tracking-[0.1em] self-center" style={{ color: SEVERITY[BANDS[i]].color }}>
              is {BANDS[i]}
            </span>,
            ...row.map((v, j) => {
              const share = v / total;
              const rgb = i === j ? '45,212,191' : j < i ? '239,68,68' : '245,158,11';
              return (
                <Motion.div
                  key={`${i}${j}`}
                  initial={{ opacity: 0, scale: 0.7 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: (i * 3 + j) * 0.05, duration: 0.35 }}
                  className="h-16 rounded-[2px] border flex flex-col items-center justify-center"
                  style={{ background: `rgba(${rgb},${0.08 + share * 0.6})`, borderColor: `rgba(${rgb},${0.25 + share * 0.5})` }}
                >
                  <span className="font-mono text-base font-semibold text-white leading-none">
                    {asPct ? `${Math.round(share * 100)}%` : v}
                  </span>
                  <span className="font-mono text-[9px] text-slate-400 mt-1">{asPct ? v : `${Math.round(share * 100)}%`}</span>
                </Motion.div>
              );
            }),
          ];
        })}
      </div>
    </InstrumentPanel>
  );
}

function SeverityTable({ rows }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="font-mono text-[10px] uppercase tracking-[0.1em] text-slate-500 border-b border-slate-700">
            <th className="text-left py-2 pr-3 font-normal">Approach</th>
            <th className="text-right py-2 px-2 font-normal">Potholes</th>
            <th className="text-right py-2 px-2 font-normal">Right</th>
            <th className="text-right py-2 px-2 font-normal">Safety cost ↓</th>
            <th className="text-right py-2 px-2 font-normal">Under-reported ↓</th>
            <th className="text-right py-2 pl-2 font-normal">Deep caught</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const ours = r.name.startsWith('Depth network');
            return (
              <tr key={r.name} className={`border-b border-slate-800 ${ours ? 'text-amber-200' : 'text-slate-300'}`}>
                <td className="py-2 pr-3">{r.name}</td>
                <td className="text-right px-2 font-mono">{r.n}</td>
                <td className="text-right px-2 font-mono">{(r.correct * 100).toFixed(1)}%</td>
                <td className="text-right px-2 font-mono">{r.cost.toFixed(2)}</td>
                <td className="text-right px-2 font-mono">{r.underReported}</td>
                <td className="text-right pl-2 font-mono">
                  {r.deepFound}/{r.nDeep} ({Math.round((r.deepFound / r.nDeep) * 100)}%)
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function SeverityShowdown({ oldBlock, newBlock, table }) {
  const [asPct, setAsPct] = useState(true);
  return (
    <InstrumentPanel
      title="Severity verdicts · old pipeline vs depth network"
      accent="red"
      flicker={false}
      headerRight={
        <div className="flex gap-1.5">
          <Chip active={asPct} onClick={() => setAsPct(true)}>Row %</Chip>
          <Chip active={!asPct} onClick={() => setAsPct(false)} tone="cyan">Counts</Chip>
        </div>
      }
      bodyClassName="p-4 space-y-4"
    >
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_1fr] gap-4 items-center">
        {oldBlock && (
          <ConfusionGrid title="Old served verdict" subtitle={`${oldBlock.n} measured potholes`} block={oldBlock} asPct={asPct} accent="red" />
        )}
        <ArrowRight className="w-6 h-6 text-amber-400 mx-auto hidden lg:block" />
        {newBlock && (
          <ConfusionGrid
            title="Depth network (served)"
            subtitle={`${newBlock.n} held-out potholes, segmenter outlines${newBlock.missed ? ` · ${newBlock.missed} not found` : ''}`}
            block={newBlock}
            asPct={asPct}
            accent="amber"
          />
        )}
      </div>
      <div className="flex flex-wrap gap-4 font-mono text-[10px] text-slate-500">
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 bg-teal-400/60" /> right band</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 bg-amber-500/60" /> over-reported (wasted repair)</span>
        <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 bg-red-500/60" /> under-reported (missed danger)</span>
      </div>
      {table && <SeverityTable rows={table} />}
      <Explain>
        Rows are what the RealSense camera measured, columns what the system said. The old verdict piled most potholes
        into Deep, including two-thirds of the measured-Shallow ones. The safety cost counts an under-report three times
        an over-report, and the depth network's cut-offs were chosen on held-out data to minimise it. The image
        classifier row is a ResNet-18 given the same measured labels and split, for a like-for-like comparison.
      </Explain>
    </InstrumentPanel>
  );
}

/* ── 6. How we got here ──────────────────────────────────────────────────── */

function VersionTip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div style={TOOLTIP_STYLE} className="px-3 py-2 max-w-[260px]">
      <p className="text-amber-300">{p.version}</p>
      <p className="text-slate-400 mb-1">{p.what}</p>
      <p>human outline {p.maeHuman} mm</p>
      <p>segmenter outline {p.maeSegmenter} mm</p>
    </div>
  );
}

function Journey({ rows }) {
  return (
    <InstrumentPanel title="How the depth network improved" accent="amber" flicker={false} bodyClassName="p-4">
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={rows} margin={{ top: 10, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#29303f" vertical={false} />
          <XAxis dataKey="version" tick={TICK} tickLine={false} axisLine={false} />
          <YAxis domain={[7, 10]} ticks={[7, 8, 9, 10]} tick={TICK} tickLine={false} axisLine={false} width={40}
            label={{ value: 'MEAN ERROR (MM)', fill: '#626b7d', fontSize: 9, fontFamily: MONO, angle: -90, position: 'insideLeft' }} />
          <Tooltip content={<VersionTip />} />
          <Legend wrapperStyle={{ color: '#97a0b1', fontSize: 11, fontFamily: MONO }} />
          <Line type="monotone" dataKey="maeHuman" name="Human outline" stroke="#f59e0b" strokeWidth={2.5} dot={{ r: 4 }} activeDot={{ r: 6 }} animationDuration={1400} />
          <Line type="monotone" dataKey="maeSegmenter" name="Segmenter outline" stroke="#22d3ee" strokeWidth={2} strokeDasharray="5 4" dot={{ r: 3 }} animationDuration={1400} />
        </LineChart>
      </ResponsiveContainer>
      <Explain>
        Mean error on the same 208 held-out potholes after each change. The biggest single step was fitting the road
        plane exactly (v4). v4 alone is marginally better here; v8+v3 is served because adding RSRD driving data cut
        the error on RSRD's held-out drives while staying inside the PothRGBD limit.
      </Explain>
    </InstrumentPanel>
  );
}

/* ── 7. Other datasets, segmenter, water ─────────────────────────────────── */

function Sources({ rows }) {
  return (
    <InstrumentPanel title="Beyond the training camera" accent="cyan" flicker={false} bodyClassName="p-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {rows.map((s, i) => (
          <Reveal key={s.dataset} delay={i * 0.07}>
            <div className="h-full bg-slate-950/60 border border-slate-700 rounded-[2px] p-3 hover:border-cyan-500/50 transition-colors">
              <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-cyan-300">{s.dataset}</p>
              <p className="font-mono text-2xl font-semibold text-white mt-1">
                <CountUp value={s.value} decimals={s.unit === 'mm' ? 1 : 2} />
                <span className="text-xs text-slate-500 ml-1">{s.unit}</span>
              </p>
              <p className="text-[11px] text-slate-400 mt-1 leading-snug">{s.metric}</p>
              <p className="font-mono text-[9px] uppercase tracking-[0.1em] text-slate-600 mt-1.5">{s.sensor}</p>
            </div>
          </Reveal>
        ))}
      </div>
      <Explain>
        Measured on data from other cameras and countries. A second camera is still the weak point: on Fan's stereo
        set the network reads deep holes too shallow, which is why the dashboard warns when a photo looks unlike the
        training photos.
      </Explain>
    </InstrumentPanel>
  );
}

function Segmenter({ rows }) {
  const data = rows.map((r) => ({ ...r, v2: +(r.foundV2 * 100).toFixed(1), v3: +(r.foundV3 * 100).toFixed(1) }));
  return (
    <InstrumentPanel title="Pothole outlines · segmenter v2 → v3" accent="holo" flicker={false} bodyClassName="p-4">
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data} margin={{ top: 16, right: 8, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#29303f" vertical={false} />
          <XAxis dataKey="set" tick={{ ...TICK, fontSize: 9 }} tickLine={false} axisLine={false} tickFormatter={(v) => v.replace(' (held-out country)', '*')} />
          <YAxis domain={[0, 100]} tick={TICK} tickLine={false} axisLine={false} tickFormatter={(v) => `${v}%`} />
          <Tooltip cursor={{ fill: 'rgba(255,255,255,0.03)' }} contentStyle={TOOLTIP_STYLE} formatter={(v) => `${v}%`} />
          <Legend wrapperStyle={{ color: '#97a0b1', fontSize: 11, fontFamily: MONO }} />
          <Bar dataKey="v2" name="v2 found" fill="#475569" radius={[2, 2, 0, 0]} animationDuration={900} />
          <Bar dataKey="v3" name="v3 found (served)" fill="#a78bfa" radius={[2, 2, 0, 0]} animationDuration={1200}>
            <LabelList dataKey="v3" position="top" formatter={(v) => `${v}%`} style={{ fill: '#cdd3dd', fontSize: 9, fontFamily: MONO }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="grid grid-cols-2 gap-2 mt-2">
        {rows.map((r) => (
          <p key={r.set} className="font-mono text-[10px] text-slate-500">
            {r.set.replace(' (held-out country)', '*')} false alarms{' '}
            <span className="text-slate-400">{r.falseV2}</span> → <span className="text-violet-300">{r.falseV3}</span>
          </p>
        ))}
      </div>
      <Explain>
        Share of potholes outlined (overlap ≥ 0.5). v3 trains on 6,306 images: v2&apos;s 1,586, plus 2,320 dashcam and
        phone frames whose boxes SAM 2 turned into outlines, plus 2,400 road frames with no pothole. *Japan was held out
        entirely: for it the numbers are boxes found and the share of pothole-free frames with a false outline.
      </Explain>
    </InstrumentPanel>
  );
}

function Water({ w }) {
  const rows = [
    { name: 'Learned water segmenter (served)', ...w.segmenter, tone: 'text-cyan-300' },
    { name: 'Hand-tuned cue ensemble (old)', ...w.ensemble, tone: 'text-slate-400' },
  ];
  return (
    <InstrumentPanel title="Water in potholes" accent="cyan" flicker={false} bodyClassName="p-4">
      <div className="flex items-center gap-2 mb-3 text-slate-400 text-xs">
        <Droplets className="w-4 h-4 text-cyan-400" /> {w.n} team-labelled potholes, {w.nWater} holding water
      </div>
      {rows.map((r) => (
        <div key={r.name} className="flex items-center justify-between border-b border-slate-800 py-2 text-xs">
          <span className={r.tone}>{r.name}</span>
          <span className="font-mono text-slate-300">
            missed <span className="text-red-400">{r.missed_water}</span> · false alarms{' '}
            <span className="text-amber-300">{r.false_alarms}</span>
          </span>
        </div>
      ))}
      <Explain>Scored by cross-fitting, so no pothole judges a threshold it helped choose.</Explain>
    </InstrumentPanel>
  );
}

/* ── 8. Why the old approaches were dropped ──────────────────────────────── */

function Counterexamples({ cards, apiBase }) {
  const [open, setOpen] = useState(null);
  return (
    <InstrumentPanel title="Why the old approaches were dropped · one counterexample each" accent="red" flicker={false} bodyClassName="p-4">
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {cards.map((c, i) => (
          <Reveal key={c.url} delay={i * 0.05}>
            <button
              type="button"
              onClick={() => setOpen(c)}
              className="group w-full h-full text-left bg-slate-950/60 border border-slate-700 rounded-[2px] p-2.5 hover:border-red-500/50 transition-colors"
            >
              <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-red-300 mb-2 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-red-500" /> {c.title}
              </p>
              <div className="overflow-hidden rounded-[2px] bg-white">
                <img src={`${apiBase}${c.url}`} alt={c.title} loading="lazy"
                  className="w-full h-44 object-contain group-hover:scale-[1.03] transition-transform duration-300" />
              </div>
              <p className="text-xs text-slate-300 mt-2 leading-snug">{c.caption}</p>
              {c.nowMm != null && (
                <p className="font-mono text-[10px] text-amber-300 mt-1.5">
                  Depth network now: {c.nowMm.toFixed(0)} mm (measured {c.measuredMm.toFixed(0)} mm)
                </p>
              )}
            </button>
          </Reveal>
        ))}
      </div>
      {open && (
        <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-4" onClick={() => setOpen(null)}>
          <InstrumentPanel
            title={open.title}
            accent="red"
            className="w-full max-w-6xl"
            headerRight={
              <button onClick={() => setOpen(null)} aria-label="Close"
                className="p-1.5 rounded-[2px] border border-slate-700 text-slate-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            }
            bodyClassName="p-4"
          >
            <img src={`${apiBase}${open.url}`} alt={open.title} className="w-full max-h-[72vh] object-contain bg-black" />
            <p className="text-sm text-slate-300 mt-3">{open.caption}</p>
          </InstrumentPanel>
        </div>
      )}
    </InstrumentPanel>
  );
}

/* ── Page ───────────────────────────────────────────────────────────────── */

export default function CurrentModelInsights({ current, apiBase }) {
  if (!current || !current.headline) {
    return (
      <InstrumentPanel title="Depth network" accent="amber" bodyClassName="p-5">
        <p className="text-sm text-slate-400">
          No results for the depth network were found. The backend reads ml_models/metric/relief_meta.json and the
          evaluation files in ml_results/.
        </p>
      </InstrumentPanel>
    );
  }
  const c = current;
  return (
    <section className="space-y-6">
      <Headline h={c.headline} table={c.severityTable} />

      {c.scatter?.length > 0 && (
        <Reveal><PredictedVsMeasured points={c.scatter} h={c.headline} /></Reveal>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        {c.baselines?.length > 0 && <Reveal><Leaderboard rows={c.baselines} /></Reveal>}
        {c.signals?.length > 0 && <Reveal delay={0.08}><Signals rows={c.signals} /></Reveal>}
      </div>

      {(c.severityOld || c.severityNew) && (
        <Reveal><SeverityShowdown oldBlock={c.severityOld} newBlock={c.severityNew} table={c.severityTable} /></Reveal>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        {c.progression?.length > 0 && <Reveal><Journey rows={c.progression} /></Reveal>}
        {c.sources?.length > 0 && <Reveal delay={0.08}><Sources rows={c.sources} /></Reveal>}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-[3fr_2fr] gap-5">
        {c.segmenter?.length > 0 && <Reveal><Segmenter rows={c.segmenter} /></Reveal>}
        {c.water && <Reveal delay={0.08}><Water w={c.water} /></Reveal>}
      </div>

      {c.counterexamples?.length > 0 && (
        <Reveal><Counterexamples cards={c.counterexamples} apiBase={apiBase} /></Reveal>
      )}

      <p className="text-center font-mono text-[10px] uppercase tracking-[0.12em] text-slate-600 max-w-3xl mx-auto leading-relaxed pb-2">
        <Layers className="w-3 h-3 inline mr-1 -mt-0.5" />
        Every number on this page is scored against measured depth on data held out from training ·{' '}
        <span className="whitespace-nowrap">fitted {c.headline.fitted}</span> ·{' '}
        <span className="whitespace-nowrap">weights {c.headline.weights?.join(' + ')}</span>
      </p>
    </section>
  );
}
