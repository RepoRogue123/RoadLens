import { CheckCircle2, MinusCircle, AlertTriangle } from 'lucide-react';

/**
 * Reports which analysis witnesses actually ran for this scan.
 *
 * Every optional backend module sits behind an import gate and a broad
 * exception handler, so a missing dependency used to disable a whole phase
 * silently while the report still looked complete. A confident answer with an
 * absent witness is worse than a visibly degraded one — so degraded state is
 * now shown rather than hidden.
 */

const LABELS = {
  geometry: 'Geometry',
  semantic: 'Semantic',
  water: 'Water',
  temporal: 'Temporal',
};

function StatusPill({ id, mod }) {
  const ran = Boolean(mod?.ran);
  const available = Boolean(mod?.available);

  // ran → green · available but didn't run → neutral · unavailable → amber warning
  const tone = ran
    ? { color: '#22c55e', Icon: CheckCircle2, text: 'OK' }
    : available
      ? { color: '#626b7d', Icon: MinusCircle, text: 'N/A' }
      : { color: '#f59e0b', Icon: AlertTriangle, text: 'OFFLINE' };

  const { Icon } = tone;
  const uncalibrated = id === 'semantic' && ran && mod?.calibrated === false;

  return (
    <span
      className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.12em]"
      title={mod?.detail || ''}
    >
      <Icon className="w-3 h-3 shrink-0" style={{ color: tone.color }} />
      <span className="text-slate-400">{LABELS[id] || id}</span>
      <span style={{ color: tone.color }}>{tone.text}</span>
      {uncalibrated && (
        <span
          className="px-1 rounded-[2px] border text-[9px]"
          style={{ color: '#fbbf24', borderColor: 'rgba(245,158,11,0.4)' }}
          title="Threshold is a hand-set default, not fitted to labelled data"
        >
          UNCALIBRATED
        </span>
      )}
    </span>
  );
}

export default function ModuleStatusStrip({ moduleStatus }) {
  if (!moduleStatus) return null;

  const order = ['geometry', 'semantic', 'water', 'temporal'];
  const degraded = order.filter((k) => moduleStatus[k] && !moduleStatus[k].available);

  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-4 py-2.5 bg-slate-950/60 border border-slate-700 rounded-[2px]">
      <span className="font-mono text-[9px] uppercase tracking-[0.2em] text-slate-600">
        Witnesses
      </span>
      {order.map((k) =>
        moduleStatus[k] ? <StatusPill key={k} id={k} mod={moduleStatus[k]} /> : null,
      )}
      {degraded.length > 0 && (
        <span className="font-mono text-[9px] uppercase tracking-[0.12em] text-amber-400 ml-auto">
          Report degraded — {degraded.length} witness{degraded.length > 1 ? 'es' : ''} unavailable
        </span>
      )}
    </div>
  );
}
