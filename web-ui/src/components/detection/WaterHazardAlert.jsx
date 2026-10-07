import { Droplets, AlertTriangle } from 'lucide-react';
import InstrumentPanel from '../instrument/InstrumentPanel';
import StatReadout from '../instrument/StatReadout';
import RadarPulse from '../instrument/RadarPulse';

export default function WaterHazardAlert({ waterAnalysis }) {
  if (!waterAnalysis?.waterDetected) return null;
  // When the learned segmenter decides, the number is the share of the pothole it marks
  // as water and the band is how much is covered. Neither is a certainty.
  const bySegmenter = waterAnalysis.combination === 'learned_segmenter';

  return (
    <InstrumentPanel
      title="Water Hazard"
      accent="red"
      statusLabel="PRIORITY ALERT"
      className="hazard-pulse"
    >
      <div className="h-[3px] hazard-stripes-red stripe-scroll" aria-hidden="true" />
      <div className="p-4 sm:p-5 flex flex-col sm:flex-row items-start sm:items-center gap-5">
        <div className="p-3 rounded-[2px] bg-red-500/15 border border-red-500/40 text-red-500 shrink-0">
          <AlertTriangle className="w-6 h-6 animate-pulse" strokeWidth={1.8} />
        </div>

        <div className="flex-1 min-w-0">
          <h3 className="text-base font-bold text-red-500 flex items-center gap-2">
            Standing water detected in this pothole
            <Droplets className="w-4 h-4 text-cyan-400 shrink-0" />
          </h3>
          <p className="text-sm text-slate-400 mt-1.5 leading-relaxed">
            Water mirrors the sky, so the interior depth reading is not trustworthy and the
            severity verdict may understate this pothole. The cross-section below switches to
            a predicted submerged floor extrapolated from the visible walls.
          </p>
        </div>

        <div className="flex items-center gap-6 shrink-0">
          <StatReadout
            label={bySegmenter ? 'Water coverage' : 'Probability'}
            value={`${(waterAnalysis.waterProbability * 100).toFixed(0)}%`}
            color="#ef4444"
          />
          <StatReadout
            label={bySegmenter ? 'Extent' : 'Confidence'}
            value={(waterAnalysis.confidenceLevel || '—').toUpperCase()}
            color="#fbbf24"
            size="sm"
          />
          {waterAnalysis.inconsistencyScore > 0.6 && (
            <StatReadout
              /* Water cue 6: boundary curvature says crater, interior depth says flat. */
              label="Depth inconsistency"
              value={waterAnalysis.inconsistencyScore.toFixed(2)}
              color="#f97316"
              size="sm"
            />
          )}
          <RadarPulse size={64} color="#ef4444" />
        </div>
      </div>
    </InstrumentPanel>
  );
}
