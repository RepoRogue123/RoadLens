import { useState } from 'react';
import { AnimatePresence, motion as Motion } from 'motion/react';
import { Radar, Sparkles, Archive } from 'lucide-react';
import InstrumentPanel from './instrument/InstrumentPanel';
import CurrentModelInsights from './insights/CurrentModelInsights';
import LegacyInsights from './insights/LegacyInsights';
import { sectionSwap } from '../theme/motion';

const VIEWS = [
  {
    id: 'current',
    icon: Sparkles,
    label: 'Current model',
    sub: 'Depth network · scored against measured depth',
  },
  {
    id: 'legacy',
    icon: Archive,
    label: 'Legacy model',
    sub: 'Classifier vote · scored against its own labels',
  },
];

/**
 * Insights: the current depth network and the retired classifier vote, kept apart
 * so the two kinds of score are never read side by side as if comparable.
 */
export default function InsightsHub({ data, loading, error, apiBase }) {
  const [view, setView] = useState('current');

  if (loading) {
    return (
      <InstrumentPanel title="Model Diagnostics" accent="cyan" statusLabel="LOADING" bodyClassName="p-5">
        <div className="flex items-center gap-2 mb-4 font-mono text-[11px] uppercase tracking-[0.14em] text-cyan-300">
          <Radar className="w-4 h-4 animate-spin" />
          Pulling benchmark artifacts…
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="shimmer rounded-[2px] h-36" />
          <div className="shimmer rounded-[2px] h-36" />
          <div className="shimmer rounded-[2px] h-36" />
        </div>
      </InstrumentPanel>
    );
  }

  if (error) {
    return (
      <InstrumentPanel title="Model Diagnostics" accent="red" statusLabel="LINK FAULT" bodyClassName="p-5">
        <p className="text-sm text-red-500">{error}</p>
        <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-slate-500 mt-2">
          Start the backend API and ensure ml_results exists.
        </p>
      </InstrumentPanel>
    );
  }

  const legacyMeasured = data?.current?.severityTable?.find((r) => r.name.startsWith('Old'));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 p-1 bg-slate-950/70 border border-slate-700 rounded-[2px]">
        {VIEWS.map((v) => {
          const active = view === v.id;
          return (
            <button
              key={v.id}
              type="button"
              onClick={() => setView(v.id)}
              className="relative text-left px-4 py-3 rounded-[2px] transition-colors"
            >
              {active && (
                <Motion.span
                  layoutId="insights-view"
                  className={`absolute inset-0 rounded-[2px] border ${
                    v.id === 'current' ? 'bg-amber-500/10 border-amber-500/50' : 'bg-slate-700/30 border-slate-500/50'
                  }`}
                  transition={{ type: 'spring', stiffness: 420, damping: 34 }}
                />
              )}
              <span className="relative flex items-center gap-2">
                <v.icon className={`w-4 h-4 ${active ? (v.id === 'current' ? 'text-amber-300' : 'text-slate-300') : 'text-slate-600'}`} />
                <span className={`font-mono text-xs uppercase tracking-[0.14em] ${active ? 'text-white' : 'text-slate-500'}`}>
                  {v.label}
                </span>
              </span>
              <span className={`relative block text-[11px] mt-0.5 ${active ? 'text-slate-400' : 'text-slate-600'}`}>{v.sub}</span>
            </button>
          );
        })}
      </div>

      <AnimatePresence mode="wait">
        <Motion.div key={view} {...sectionSwap}>
          {view === 'current' ? (
            <CurrentModelInsights current={data?.current} apiBase={apiBase} />
          ) : (
            <LegacyInsights data={data} apiBase={apiBase} measured={legacyMeasured} />
          )}
        </Motion.div>
      </AnimatePresence>
    </div>
  );
}
