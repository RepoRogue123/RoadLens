import { useState } from 'react';
import { motion as Motion } from 'motion/react';
import InstrumentPanel from '../instrument/InstrumentPanel';
import { responseFor } from '../../utils/wheelDrop';

const CATEGORY_COLORS = {
  Observation: '#10b981', D3: '#f59e0b', D3A: '#f97316', D2: '#ef4444', D1: '#b91c1c',
};
const LIKELIHOOD = ['Rare', 'Unlikely', 'Possible', 'Likely', 'Almost certain'];
const CONSEQUENCE = ['Insignificant', 'Minor', 'Moderate', 'Significant', 'Major'];

// Where the pothole is decides the likelihood; the photo cannot show it, so the user sets it.
const LIKELIHOOD_HINT = [
  'Quiet road, pothole outside the wheel path',
  'Low traffic, near the edge',
  'Ordinary road, partly in the wheel path',
  'Busy road, in the wheel path',
  'Main route, in the wheel path of most vehicles',
];

// The parent keys this panel by pothole, so a new scan starts from the server's values.
export default function RiskMatrixPanel({ risk }) {
  const [likelihood, setLikelihood] = useState(risk?.likelihood || 3);
  const [route, setRoute] = useState(risk?.route || 'local');
  if (!risk) return null;

  const c = risk.consequence;
  const score = likelihood * c;
  const res = responseFor(score, route);
  const color = CATEGORY_COLORS[res.category] || '#64748b';
  const learned = risk.consequence_source === 'learned';

  return (
    <InstrumentPanel title="Repair Priority · UK Risk Matrix" accent="amber" statusLabel={learned ? 'LEARNED CONSEQUENCE' : 'PROVISIONAL CONSEQUENCE'}
                     bodyClassName="p-4 sm:p-5" flicker={false}>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div>
          <Motion.div key={res.category + route} initial={{ scale: 0.92, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                      className="rounded-[2px] px-4 py-3 text-center" style={{ background: color }}>
            <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-white/80">Category {res.category}</div>
            <div className="text-lg font-bold text-white">Respond within {res.response}</div>
          </Motion.div>
          <div className="mt-4 grid grid-cols-[auto_1fr] gap-x-2 items-end">
            <div className="flex flex-col-reverse justify-between h-full pb-6">
              {[1, 2, 3, 4, 5].map((l) => (
                <span key={l} className="font-mono text-[10px] text-slate-500 h-8 flex items-center">{l}</span>
              ))}
            </div>
            <div>
              <div className="grid grid-cols-5 gap-[3px]">
                {[5, 4, 3, 2, 1].map((l) => [1, 2, 3, 4, 5].map((ci) => {
                  const here = l === likelihood && ci === c;
                  const cat = responseFor(l * ci, route).category;
                  return (
                    <button key={`${l}-${ci}`} type="button" onClick={() => setLikelihood(l)}
                            className="h-8 rounded-[1px] font-mono text-[11px] text-white transition-all"
                            style={{ background: CATEGORY_COLORS[cat], opacity: here ? 1 : 0.28,
                                     outline: here ? '2px solid white' : 'none', fontWeight: here ? 700 : 400 }}
                            title={`likelihood ${l} x consequence ${ci} = ${l * ci} (${cat})`}>
                      {l * ci}
                    </button>
                  );
                }))}
              </div>
              <div className="grid grid-cols-5 mt-1">
                {[1, 2, 3, 4, 5].map((ci) => (
                  <span key={ci} className={`text-center font-mono text-[10px] ${ci === c ? 'text-white' : 'text-slate-500'}`}>{ci}</span>
                ))}
              </div>
              <div className="text-center font-mono text-[10px] uppercase tracking-[0.12em] text-slate-500 mt-1">consequence</div>
            </div>
          </div>
        </div>

        <div className="space-y-4 text-sm">
          <div className="bg-slate-950/60 border border-slate-700 rounded-[2px] p-3">
            <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500">Consequence · from the pothole</div>
            <div className="text-white font-semibold mt-1">{c} · {CONSEQUENCE[c - 1]}</div>
            <p className="text-xs text-slate-400 mt-1 leading-relaxed">
              {learned
                ? risk.consequence_rule
                : `Chance it reaches the ${risk.investigatory_level.depth_mm} mm investigatory depth: ${(100 * risk.p_investigatory).toFixed(0)}%. Provisional rule until the team's ratings train the learned version.`}
            </p>
          </div>
          <div className="bg-slate-950/60 border border-slate-700 rounded-[2px] p-3">
            <div className="font-mono text-[10px] uppercase tracking-[0.14em] text-slate-500">Likelihood · where it is (you set this)</div>
            <div className="flex gap-1 mt-2">
              {[1, 2, 3, 4, 5].map((l) => (
                <button key={l} type="button" onClick={() => setLikelihood(l)}
                        className={`flex-1 py-1 rounded-[2px] border font-mono text-xs ${l === likelihood ? 'border-amber-400 text-amber-300 bg-amber-500/10' : 'border-slate-700 text-slate-400'}`}>
                  {l}
                </button>
              ))}
            </div>
            <div className="text-white mt-2">{LIKELIHOOD[likelihood - 1]}</div>
            <p className="text-xs text-slate-500">{LIKELIHOOD_HINT[likelihood - 1]}</p>
            <div className="flex gap-1 mt-3">
              {[['local', 'Local route'], ['county', 'County route']].map(([k, label]) => (
                <button key={k} type="button" onClick={() => setRoute(k)}
                        className={`flex-1 py-1 rounded-[2px] border font-mono text-[11px] ${route === k ? 'border-cyan-400 text-cyan-300' : 'border-slate-700 text-slate-400'}`}>
                  {label}
                </button>
              ))}
            </div>
          </div>
          <p className="text-[11px] text-slate-500 leading-relaxed">
            Risk = likelihood × consequence, with the categories and response times of the UK code of practice
            &ldquo;Well-managed Highway Infrastructure&rdquo; as published by Essex Highways. Score {likelihood} × {c} = {score}.
          </p>
        </div>
      </div>
    </InstrumentPanel>
  );
}
