import { useState } from 'react';
import { severityToken, normalizeSeverity } from '../theme/severity';
import { CLASSIFIERS } from '../theme/classifiers';

/**
 * Ledger of per-classifier verdicts. Shows each model's severity and whether
 * it matches the consensus — no fabricated confidence numbers.
 */
export default function ClassifierTable({ results, consensus }) {
  const [hoveredRow, setHoveredRow] = useState(null);
  const consensusKey = normalizeSeverity(consensus);

  return (
    <div className="divide-y divide-slate-700/50">
      {CLASSIFIERS.map((cls) => {
        const result = results[cls.id];
        if (!result) return null;
        const token = severityToken(result.severity);
        const matches = normalizeSeverity(result.severity) === consensusKey;
        const isRule = cls.type === 'rule';
        const isMeasured = cls.type === 'measured';

        return (
          <div
            key={cls.id}
            onMouseEnter={() => setHoveredRow(cls.id)}
            onMouseLeave={() => setHoveredRow(null)}
            className={`relative flex items-center px-4 sm:px-5 py-3 transition-colors duration-150 ${
              hoveredRow === cls.id ? 'bg-white/[0.03]' : ''
            }`}
          >
            {/* Classifier name */}
            <div className="flex-1 min-w-0 flex items-center gap-2">
              <span
                className={`text-sm font-medium truncate ${
                  isRule ? 'font-mono text-cyan-300' : 'text-slate-300'
                }`}
              >
                {cls.name}
              </span>
              {isRule && (
                <span className="font-mono text-[9px] px-1.5 py-0.5 rounded-[2px] bg-cyan-500/10 text-cyan-400 border border-cyan-500/25 tracking-[0.1em] shrink-0">
                  THRESHOLD
                </span>
              )}
              {isMeasured && (
                <span className="font-mono text-[9px] px-1.5 py-0.5 rounded-[2px] bg-green-500/10 text-green-400 border border-green-500/25 tracking-[0.1em] shrink-0">
                  MEASURED
                </span>
              )}
            </div>

            <div className="flex items-center gap-4 sm:gap-6 shrink-0">
              {/* Severity pill */}
              <span
                className="px-2.5 py-1 rounded-[2px] font-mono text-[10px] uppercase tracking-[0.1em] font-semibold border"
                style={{
                  backgroundColor: token.bg,
                  color: token.color,
                  borderColor: token.ring,
                }}
              >
                {token.label}
              </span>

              {/* Consensus alignment */}
              <span
                className={`font-mono text-[10px] tracking-[0.12em] w-[72px] text-right ${
                  matches ? 'text-green-500' : 'text-amber-400'
                }`}
              >
                {matches ? 'MATCH' : 'DIVERGES'}
              </span>
            </div>

            {/* Hover note */}
            {hoveredRow === cls.id && (
              <div className="absolute -top-8 left-1/2 -translate-x-1/2 z-10 px-3 py-1.5 rounded-[2px] bg-slate-800 border border-slate-600 font-mono text-[10px] text-slate-300 whitespace-nowrap shadow-xl">
                {isMeasured
                  ? 'Trained on 1,051 potholes with RealSense-measured depth'
                  : isRule
                    ? 'Hand-set thresholds on relative depth contrast'
                    : 'Trained on clustered pseudo-labels, not measurement'}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
