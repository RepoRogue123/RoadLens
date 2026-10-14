import { useMemo, useState } from 'react';
import { Filter, Expand, X, ChevronLeft, ChevronRight, Archive } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import InstrumentPanel from '../instrument/InstrumentPanel';
import StatReadout from '../instrument/StatReadout';
import { SEVERITY } from '../../theme/severity';

const MODEL_COLORS = ['#f59e0b', '#22d3ee', '#a78bfa', '#2dd4bf', '#fcd34d', '#ea580c'];
const SEVERITY_COLORS = {
  Shallow: SEVERITY.Shallow.color,
  Moderate: SEVERITY.Moderate.color,
  Deep: SEVERITY.Deep.color,
  Unknown: SEVERITY.Unknown.color,
};

const MONO_TICK = { fill: '#97a0b1', fontSize: 10, fontFamily: 'IBM Plex Mono, monospace' };

function niceModelName(name) {
  if (!name) return 'Unknown';
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (s) => s.toUpperCase());
}

function chipClass(active, tone = 'amber') {
  const activeTone =
    tone === 'cyan'
      ? 'border-cyan-500/40 text-cyan-300 bg-cyan-500/10'
      : 'border-amber-500/40 text-amber-300 bg-amber-500/10';
  return `px-2.5 py-1 rounded-[2px] border font-mono text-[10px] uppercase tracking-[0.1em] transition-colors ${
    active ? activeTone : 'border-slate-700 text-slate-500 hover:text-slate-300'
  }`;
}

function MetricTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;
  const row = payload[0].payload;
  return (
    <div className="bg-slate-950/95 border border-slate-600 rounded-[2px] px-4 py-3 shadow-xl max-w-[260px] font-mono">
      <p className="text-sm font-semibold text-white mb-1">{niceModelName(label)}</p>
      <p className="text-xs text-slate-400">
        Accuracy <span className="text-amber-300">{(row.accuracy ?? 0).toFixed(2)}%</span>
      </p>
      <p className="text-xs text-slate-400">
        Macro F1 <span className="text-cyan-300">{(row.macroF1 ?? 0).toFixed(2)}%</span>
      </p>
    </div>
  );
}

function ScatterTooltip({ active, payload }) {
  if (!active || !payload || payload.length === 0) return null;
  const p = payload[0].payload;
  return (
    <div className="bg-slate-950/95 border border-slate-600 rounded-[2px] px-4 py-3 shadow-xl max-w-[280px] font-mono">
      <p className="text-sm font-semibold text-white">{p.imageName || 'Sample'}</p>
      <p className="text-xs text-slate-400 mt-1">
        Severity <span style={{ color: SEVERITY_COLORS[p.severity] || SEVERITY_COLORS.Unknown }}>{p.severity}</span>
      </p>
      <p className="text-xs text-slate-500 mt-1">X {Number(p.x).toFixed(4)}</p>
      <p className="text-xs text-slate-500">Y {Number(p.y).toFixed(4)}</p>
    </div>
  );
}

/**
 * The retired pipeline: five classifiers voting on Depth-Anything features, trained and
 * scored on KMeans pseudo-labels. Kept for the record; `measured` is how its served
 * verdict did against measured depth, from the current-model insights.
 */
export default function LegacyInsights({ data, apiBase, measured }) {
  const [metricMode, setMetricMode] = useState('accuracy');
  const [galleryCategory, setGalleryCategory] = useState('All');
  const [gallerySearch, setGallerySearch] = useState('');
  const [activeGraphIndex, setActiveGraphIndex] = useState(null);
  const [xFeature, setXFeature] = useState('max_depth');
  const [yFeature, setYFeature] = useState('pothole_area');
  const [severityFilter, setSeverityFilter] = useState('All');

  const metrics = useMemo(() => data?.metrics ?? [], [data]);
  const ablation = useMemo(() => data?.ablation ?? [], [data]);
  const featureColumns = useMemo(() => data?.featureColumns ?? [], [data]);
  const featureRows = useMemo(() => data?.featureRows ?? [], [data]);
  const graphs = useMemo(() => data?.graphs ?? [], [data]);

  const metricRows = useMemo(() => {
    const rows = metrics.map((m) => ({
      model: m.model,
      accuracy: Number((m.accuracy ?? 0) * 100),
      macroF1: Number(((m.macroF1 ?? 0) * 100)),
    }));
    const key = metricMode === 'macroF1' ? 'macroF1' : 'accuracy';
    return rows.sort((a, b) => (b[key] ?? 0) - (a[key] ?? 0));
  }, [metrics, metricMode]);

  const ablationRows = useMemo(() => {
    return ablation.map((row) => ({
      subset: row.subset,
      valAccuracy: Number((row.valAccuracy ?? 0) * 100),
      macroF1: Number((row.macroF1 ?? 0) * 100),
      numFeatures: row.numFeatures,
    }));
  }, [ablation]);

  const severityOptions = useMemo(() => {
    const uniques = new Set(featureRows.map((r) => r.severity || 'Unknown'));
    return ['All', ...Array.from(uniques)];
  }, [featureRows]);

  const featureRowsFiltered = useMemo(() => {
    return featureRows
      .filter((row) => severityFilter === 'All' || row.severity === severityFilter)
      .slice(0, 1000)
      .map((row) => ({
        imageName: row.imageName,
        severity: row.severity || 'Unknown',
        x: Number(row[xFeature] ?? 0),
        y: Number(row[yFeature] ?? 0),
      }));
  }, [featureRows, severityFilter, xFeature, yFeature]);

  const groupedScatter = useMemo(() => {
    const map = {};
    for (const row of featureRowsFiltered) {
      const key = row.severity || 'Unknown';
      if (!map[key]) map[key] = [];
      map[key].push(row);
    }
    return map;
  }, [featureRowsFiltered]);

  const galleryCategories = useMemo(() => {
    const categories = Array.from(new Set(graphs.map((g) => g.category || 'Other')));
    return ['All', ...categories];
  }, [graphs]);

  const galleryItems = useMemo(() => {
    const query = gallerySearch.trim().toLowerCase();
    return graphs.filter((graph) => {
      const catMatch = galleryCategory === 'All' || graph.category === galleryCategory;
      const searchMatch = query.length === 0 || graph.title.toLowerCase().includes(query);
      return catMatch && searchMatch;
    });
  }, [graphs, galleryCategory, gallerySearch]);

  const activeGraph = activeGraphIndex === null ? null : galleryItems[activeGraphIndex] ?? null;

  const hasFeatures = featureColumns.length > 0 && featureRows.length > 0;

  return (
    <section className="space-y-6">
      {/* Retirement notice: these scores are circular */}
      <div className="hazard-stripes rounded-[2px] p-[3px]">
        <div className="bg-slate-950 rounded-[2px] px-4 py-3 flex gap-3">
          <Archive className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="text-sm text-slate-300 leading-relaxed">
            <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-amber-300 mb-1">
              Retired model · kept for the record
            </p>
            The scores below were measured against labels this pipeline generated itself (KMeans clusters of the same
            features), so the near-perfect accuracy is circular.
            {measured && (
              <>
                {' '}Scored against real measured depth on {measured.n} potholes, its served verdict was right{' '}
                <span className="text-red-400 font-mono">{(measured.correct * 100).toFixed(1)}%</span> of the time and
                under-reported {measured.underReported}.
              </>
            )}{' '}
            It is still computed by the API and shown as a secondary signal in each scan.
          </div>
        </div>
      </div>

      {/* Summary readouts */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <InstrumentPanel accent="amber" flicker={false} bodyClassName="p-5">
          <StatReadout label="Models benchmarked" value={metrics.length} size="lg" color="#fff" />
          <p className="font-mono text-[10px] text-slate-600 mt-2 uppercase tracking-[0.1em]">From classification reports</p>
        </InstrumentPanel>
        <InstrumentPanel accent="cyan" flicker={false} bodyClassName="p-5">
          <StatReadout label="Validation graphs" value={graphs.length} size="lg" color="#fff" />
          <p className="font-mono text-[10px] text-slate-600 mt-2 uppercase tracking-[0.1em]">Interactive gallery below</p>
        </InstrumentPanel>
        <InstrumentPanel accent="holo" flicker={false} bodyClassName="p-5">
          <StatReadout
            label="Top model"
            value={metricRows[0] ? niceModelName(metricRows[0].model) : 'N/A'}
            color="#fff"
          />
          <p className="font-mono text-[11px] text-amber-300 mt-2">
            {(metricRows[0]?.accuracy ?? 0).toFixed(2)}% ACCURACY
          </p>
        </InstrumentPanel>
      </div>

      {/* Performance + ablation */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <InstrumentPanel
          title="Model Performance Explorer"
          accent="amber"
          flicker={false}
          headerRight={
            <div className="flex items-center gap-1.5">
              <button onClick={() => setMetricMode('accuracy')} className={chipClass(metricMode === 'accuracy')}>
                Accuracy
              </button>
              <button onClick={() => setMetricMode('macroF1')} className={chipClass(metricMode === 'macroF1', 'cyan')}>
                Macro F1
              </button>
            </div>
          }
          bodyClassName="p-4"
        >
          <ResponsiveContainer width="100%" height={290}>
            <BarChart data={metricRows} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#29303f" vertical={false} />
              <XAxis
                dataKey="model"
                tick={{ ...MONO_TICK, fontSize: 10 }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => niceModelName(v).slice(0, 14)}
                angle={-15}
                textAnchor="end"
                height={52}
              />
              <YAxis
                tick={MONO_TICK}
                tickLine={false}
                axisLine={false}
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
              />
              <Tooltip content={<MetricTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              <Bar dataKey={metricMode} radius={[2, 2, 0, 0]} maxBarSize={55}>
                {metricRows.map((_, idx) => (
                  <Cell key={idx} fill={MODEL_COLORS[idx % MODEL_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </InstrumentPanel>

        <InstrumentPanel
          title="Ablation Study"
          accent="cyan"
          statusLabel="ACC VS F1"
          flicker={false}
          bodyClassName="p-4"
        >
          <ResponsiveContainer width="100%" height={290}>
            <BarChart data={ablationRows} margin={{ top: 8, right: 10, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#29303f" vertical={false} />
              <XAxis dataKey="subset" tick={MONO_TICK} tickLine={false} axisLine={false} />
              <YAxis
                tick={MONO_TICK}
                tickLine={false}
                axisLine={false}
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
              />
              <Tooltip
                cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                contentStyle={{
                  background: 'rgba(10,12,16,0.95)',
                  border: '1px solid #3c4354',
                  borderRadius: 2,
                  fontSize: 11,
                  fontFamily: 'IBM Plex Mono, monospace',
                  color: '#cdd3dd',
                }}
              />
              <Legend wrapperStyle={{ color: '#97a0b1', fontSize: 11, fontFamily: 'IBM Plex Mono, monospace' }} />
              <Bar dataKey="valAccuracy" name="Val Accuracy" fill="#f59e0b" radius={[2, 2, 0, 0]} />
              <Bar dataKey="macroF1" name="Macro F1" fill="#22d3ee" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </InstrumentPanel>
      </div>

      {/* Feature explorer */}
      <InstrumentPanel
        title="Interactive Feature Explorer"
        accent="holo"
        flicker={false}
        headerRight={
          <div className="flex flex-wrap gap-1.5">
            {[
              { value: xFeature, set: setXFeature, cols: featureColumns.length ? featureColumns : ['max_depth'] },
              { value: yFeature, set: setYFeature, cols: featureColumns.length ? featureColumns : ['pothole_area'] },
              { value: severityFilter, set: setSeverityFilter, cols: severityOptions },
            ].map((sel, i) => (
              <select
                key={i}
                value={sel.value}
                onChange={(e) => sel.set(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-[2px] px-2 py-1 font-mono text-[10px] text-slate-300 focus:outline-none focus:border-amber-500/50"
                disabled={!hasFeatures}
              >
                {sel.cols.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            ))}
          </div>
        }
        bodyClassName="p-4"
      >
        {hasFeatures ? (
          <>
            <ResponsiveContainer width="100%" height={330}>
              <ScatterChart margin={{ top: 10, right: 20, bottom: 24, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#29303f" />
                <XAxis
                  dataKey="x"
                  name={xFeature}
                  tick={MONO_TICK}
                  tickLine={false}
                  axisLine={false}
                  label={{ value: xFeature.toUpperCase(), fill: '#626b7d', fontSize: 9, fontFamily: 'IBM Plex Mono, monospace', position: 'insideBottom', offset: -10 }}
                />
                <YAxis
                  dataKey="y"
                  name={yFeature}
                  tick={MONO_TICK}
                  tickLine={false}
                  axisLine={false}
                  label={{ value: yFeature.toUpperCase(), fill: '#626b7d', fontSize: 9, fontFamily: 'IBM Plex Mono, monospace', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip content={<ScatterTooltip />} cursor={{ strokeDasharray: '4 4' }} />
                <Legend wrapperStyle={{ color: '#97a0b1', fontSize: 11, fontFamily: 'IBM Plex Mono, monospace' }} />
                {Object.entries(groupedScatter).map(([severity, points]) => (
                  <Scatter
                    key={severity}
                    name={severity}
                    data={points}
                    fill={SEVERITY_COLORS[severity] || SEVERITY_COLORS.Unknown}
                    line={false}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
            <p className="font-mono text-[10px] uppercase tracking-[0.1em] text-slate-600 mt-2">
              {featureRowsFiltered.length} samples · {xFeature} vs {yFeature}
            </p>
          </>
        ) : (
          <p className="text-sm text-slate-400 py-6">Feature CSV files were not found in ml_results.</p>
        )}
      </InstrumentPanel>

      {/* Graph gallery */}
      <InstrumentPanel
        title={`Graph Gallery · ${galleryItems.length}/${graphs.length}`}
        accent="amber"
        flicker={false}
        headerRight={
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="flex items-center gap-1 font-mono text-[10px] text-slate-600 uppercase">
              <Filter className="w-3 h-3" /> Filter
            </span>
            {galleryCategories.map((category) => (
              <button
                key={category}
                onClick={() => {
                  setGalleryCategory(category);
                  setActiveGraphIndex(null);
                }}
                className={chipClass(galleryCategory === category)}
              >
                {category}
              </button>
            ))}
            <input
              value={gallerySearch}
              onChange={(e) => setGallerySearch(e.target.value)}
              placeholder="SEARCH"
              className="bg-slate-900 border border-slate-700 rounded-[2px] px-2 py-1 font-mono text-[10px] text-slate-300 placeholder:text-slate-600 w-24 focus:outline-none focus:border-amber-500/50"
            />
          </div>
        }
        bodyClassName="p-4"
      >
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {galleryItems.map((graph, idx) => (
            <button
              key={graph.file}
              type="button"
              onClick={() => setActiveGraphIndex(idx)}
              className="group text-left bg-slate-950/60 rounded-[2px] p-2.5 border border-slate-700 hover:border-amber-500/50 transition-all duration-200"
            >
              <div className="relative overflow-hidden rounded-[2px] bg-black">
                <img
                  src={`${apiBase}${graph.url}`}
                  alt={graph.title}
                  className="w-full h-44 object-cover group-hover:scale-[1.02] transition-transform duration-300"
                  loading="lazy"
                />
                <span className="absolute top-2 left-2 font-mono text-[9px] uppercase tracking-[0.1em] px-2 py-1 rounded-[2px] bg-black/75 text-amber-300 border border-amber-500/30">
                  {graph.category}
                </span>
                <span className="absolute top-2 right-2 p-1 rounded-[2px] bg-black/70 text-slate-300 border border-slate-700">
                  <Expand className="w-3.5 h-3.5" />
                </span>
              </div>
              <p className="text-sm text-slate-300 mt-2 min-h-[2.5rem]">{graph.title}</p>
            </button>
          ))}
        </div>

        {galleryItems.length === 0 && (
          <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-slate-500 py-8 text-center">
            No graphs match the current filters
          </p>
        )}
      </InstrumentPanel>

      {/* Fullscreen modal */}
      {activeGraph && (
        <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center p-4">
          <InstrumentPanel
            title={activeGraph.title}
            accent="cyan"
            statusLabel={`${activeGraphIndex + 1} / ${galleryItems.length}`}
            className="w-full max-w-6xl"
            headerRight={
              <button
                onClick={() => setActiveGraphIndex(null)}
                className="p-1.5 rounded-[2px] border border-slate-700 text-slate-400 hover:text-white hover:border-slate-500 transition-colors"
                aria-label="Close graph modal"
              >
                <X className="w-4 h-4" />
              </button>
            }
            bodyClassName="p-4"
          >
            <div className="relative rounded-[2px] overflow-hidden bg-black scanline-overlay">
              <img
                src={`${apiBase}${activeGraph.url}`}
                alt={activeGraph.title}
                className="w-full max-h-[70vh] object-contain"
              />
            </div>

            <div className="flex items-center justify-between mt-4">
              <button
                onClick={() => setActiveGraphIndex((prev) => (prev === null ? 0 : (prev - 1 + galleryItems.length) % galleryItems.length))}
                className="px-3 py-2 rounded-[2px] border border-slate-700 text-slate-300 font-mono text-[11px] uppercase tracking-[0.1em] hover:border-amber-500/50 transition-colors"
              >
                <span className="flex items-center gap-1"><ChevronLeft className="w-4 h-4" /> Prev</span>
              </button>
              <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-amber-300/80">
                {activeGraph.category}
              </span>
              <button
                onClick={() => setActiveGraphIndex((prev) => (prev === null ? 0 : (prev + 1) % galleryItems.length))}
                className="px-3 py-2 rounded-[2px] border border-slate-700 text-slate-300 font-mono text-[11px] uppercase tracking-[0.1em] hover:border-amber-500/50 transition-colors"
              >
                <span className="flex items-center gap-1">Next <ChevronRight className="w-4 h-4" /></span>
              </button>
            </div>
          </InstrumentPanel>
        </div>
      )}

      <p className="text-center font-mono text-[10px] uppercase tracking-[0.12em] text-slate-600 max-w-2xl mx-auto leading-relaxed pb-2">
        Metrics stream from ml_results artifacts · dependent on pseudo-label quality and dataset distribution
      </p>
    </section>
  );
}
