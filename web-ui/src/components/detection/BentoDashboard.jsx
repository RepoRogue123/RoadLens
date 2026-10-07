import { motion as Motion } from 'motion/react';
import { Target, ListChecks } from 'lucide-react';

import SeverityBadge from '../SeverityBadge';
import FeatureStrip from '../FeatureStrip';
import ClassifierTable from '../ClassifierTable';
import { NAME_TO_ID } from '../../theme/classifiers';
import InstrumentPanel from '../instrument/InstrumentPanel';
import ModuleStatusStrip from '../instrument/ModuleStatusStrip';
import InteractiveViewport from './InteractiveViewport';
import HologramTerrain from '../holo/HologramTerrain';
import DepthProfileChart from './DepthProfileChart';
import TemporalPrognosis from './TemporalPrognosis';
import SemanticIntelligence from './SemanticIntelligence';
import WaterHazardAlert from './WaterHazardAlert';
import { panelCascade, panelEnter } from '../../theme/motion';

/** Mono section rail: ── LABEL ────── */
function SectionRail({ label }) {
  return (
    <div className="flex items-center gap-3 mt-2" aria-hidden="true">
      <span className="w-4 h-px bg-slate-600" />
      <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-slate-500">{label}</span>
      <span className="flex-1 h-px bg-slate-800" />
    </div>
  );
}

export default function BentoDashboard({ results }) {
  if (!results || !results.potholes || results.potholes.length === 0) {
    return (
      <InstrumentPanel title="Scan Report" accent="cyan" statusLabel="SURFACE NOMINAL" bodyClassName="p-8 text-center">
        <p className="font-mono text-sm text-green-500 tracking-[0.14em] uppercase">
          No pothole detected in this frame
        </p>
        <p className="text-sm text-slate-500 mt-2">
          The segmentation model found no damage above the confidence threshold.
        </p>
      </InstrumentPanel>
    );
  }

  // The API's representative pothole (most severe, then largest) supplies the
  // headline verdict, features and agreement. The advanced panels must show the
  // same pothole, or the verdict and the diagnostics can describe two different
  // ones on a multi-pothole frame.
  const primaryPothole =
    results.potholes.find((p) => p.id === results.representativeId) || results.potholes[0];
  // The headline comes from the measured-label depth model when it ran; the
  // legacy vote is then shown for comparison only.
  const measured = results.severitySource === 'metric';
  const metric = primaryPothole.metricDepth || null;

  const formattedClassifiers = {};
  if (results.classifications) {
    for (const [name, verdict] of Object.entries(results.classifications)) {
      formattedClassifiers[NAME_TO_ID[name] || name] = { severity: verdict };
    }
  }

  return (
    <Motion.div variants={panelCascade} initial="hidden" animate="show" className="space-y-6">
      {/* 0. Which witnesses actually ran */}
      {results.moduleStatus && (
        <Motion.div variants={panelEnter}>
          <ModuleStatusStrip moduleStatus={results.moduleStatus} />
        </Motion.div>
      )}

      {/* 1. Priority alert */}
      <Motion.div variants={panelEnter}>
        <WaterHazardAlert waterAnalysis={primaryPothole.waterAnalysis} />
      </Motion.div>

      {/* 2. Holographic depth terrain — the signature panel */}
      <Motion.div variants={panelEnter}>
        <HologramTerrain
          images={results.images}
          primaryPothole={primaryPothole}
          features={results.features}
        />
      </Motion.div>

      {/* 3. Optical feed + consensus */}
      <Motion.div variants={panelEnter} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 h-full">
          <InteractiveViewport images={results.images} potholeCount={results.potholeCount} />
        </div>

        <InstrumentPanel
          title={measured ? 'Severity' : 'Consensus Severity'}
          accent="amber"
          statusLabel={measured ? 'MEASURED MODEL' : 'VERDICT'}
          flicker={false}
          className="lg:col-span-1"
          bodyClassName="p-6 flex flex-col items-center justify-center text-center h-full"
        >
          <SeverityBadge
            severity={results.consensusSeverity}
            consensusCount={results.consensusCount}
            total={results.totalClassifiers}
          />
          <p className="mt-4 text-slate-400 text-sm">{results.consensusSubtext}</p>
          {metric && (
            <div className="mt-3 w-full font-mono">
              <p className="text-white text-lg">
                {metric.depthMm.toFixed(0)} mm
                <span className="text-slate-500 text-xs"> · likely {metric.lowMm.toFixed(0)}–{metric.highMm.toFixed(0)} mm</span>
              </p>
              <p className="mt-1 text-[10px] uppercase tracking-[0.12em] text-slate-500">
                Estimated bowl depth · 80% of held-out potholes within ±{metric.intervalMm.toFixed(0)} mm
                {metric.readFrom === 'close-up' && ' · small in the photo, read from a close-up'}
              </p>
              {metric.inDistribution === false && (
                <p
                  className="mt-2 text-[10px] uppercase tracking-[0.12em] text-amber-400 leading-relaxed"
                  title={`Outside the training range: ${(metric.outOfRange || []).join(', ')}`}
                >
                  Photo unlike the measured training set · estimate is an extrapolation
                  {(metric.outOfRange || []).includes('mf_cam_dist_mm') && metric.trainedCameraRangeMm?.length === 2 && (
                    <span className="block text-slate-400 normal-case tracking-normal mt-1">
                      Camera looks {metric.cameraFurtherThanTraining ? 'further from' : 'closer to'} the road
                      than in the measured photos, which were taken{' '}
                      {(metric.trainedCameraRangeMm[0] / 1000).toFixed(1)}–
                      {(metric.trainedCameraRangeMm[1] / 1000).toFixed(1)} m away, looking down.
                    </span>
                  )}
                </p>
              )}
            </div>
          )}
          {measured && primaryPothole.legacyVote && primaryPothole.legacyVote !== results.consensusSeverity && (
            <p className="mt-2 font-mono text-[10px] uppercase tracking-[0.12em] text-slate-400">
              Legacy pseudo-label vote said {primaryPothole.legacyVote}
            </p>
          )}
          {primaryPothole.semanticSuggestion && (
            <p className="mt-2 font-mono text-[10px] uppercase tracking-[0.12em] text-slate-500">
              DINOv2 suggested {primaryPothole.semanticSuggestion} · advisory, not applied
            </p>
          )}
          {primaryPothole.votedSeverity && (
            <p className="mt-2 font-mono text-[10px] uppercase tracking-[0.12em] text-amber-400">
              Semantic override · classifiers voted {primaryPothole.votedSeverity}
            </p>
          )}

          <div className="w-full h-px bg-slate-700/70 my-5" />

          <div className="w-full flex justify-between items-center px-2">
            <div className="flex items-center gap-2 text-slate-400">
              <Target className="w-4 h-4 text-cyan-400" />
              <span className="font-mono text-[11px] uppercase tracking-[0.1em]">Targets</span>
            </div>
            <span className="text-xl font-bold font-mono text-white">{results.potholeCount}</span>
          </div>

          <div className="w-full flex justify-between items-center px-2 mt-3">
            <div className="flex items-center gap-2 text-slate-400">
              <ListChecks className="w-4 h-4 text-green-500" />
              <span className="font-mono text-[11px] uppercase tracking-[0.1em]">Agreement</span>
            </div>
            <span className="text-xl font-bold font-mono text-white">
              {results.consensusCount}/{results.totalClassifiers}
            </span>
          </div>
        </InstrumentPanel>
      </Motion.div>

      {/* 3. Depth cross-section */}
      <Motion.div variants={panelEnter}>
        <DepthProfileChart potholes={results.potholes} />
      </Motion.div>

      {/* 4. Core telemetry */}
      <Motion.div variants={panelEnter}>
        <SectionRail label="Core metrics" />
        <div className="mt-3">
          <FeatureStrip features={results.features} />
        </div>
      </Motion.div>

      {/* 5. Advanced diagnostics */}
      <Motion.div variants={panelEnter}>
        <SectionRail label="Advanced diagnostics" />
        <div className="mt-3 space-y-6">
          <TemporalPrognosis
            temporalAnalysis={primaryPothole.temporalAnalysis}
            currentSeverity={results.consensusSeverity}
          />
          <SemanticIntelligence geometryAnalysis={primaryPothole.geometryAnalysis} />
        </div>
      </Motion.div>

      {/* 6. Classifier ledger */}
      <Motion.div variants={panelEnter}>
        <SectionRail label="Classifier breakdown" />
        <div className="mt-3">
          <InstrumentPanel title="Classifier Ledger" accent="amber" statusLabel={`${results.totalClassifiers} MODELS`} flicker={false}>
            <ClassifierTable
              results={formattedClassifiers}
              consensus={results.consensusSeverity}
            />
          </InstrumentPanel>
        </div>
      </Motion.div>
    </Motion.div>
  );
}
