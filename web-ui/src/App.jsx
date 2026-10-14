import { useState, useCallback, useEffect, useRef, lazy, Suspense } from 'react';
import { AnimatePresence, motion as Motion } from 'motion/react';
import { RefreshCw, RotateCcw } from 'lucide-react';

import TelemetryBar from './components/instrument/TelemetryBar';
import BackgroundField from './components/instrument/BackgroundField';
import InstrumentPanel from './components/instrument/InstrumentPanel';
import UploadPanel from './components/UploadPanel';
import ScanSequence from './components/scan/ScanSequence';
import TargetLock from './components/scan/TargetLock';
import InsightsHub from './components/InsightsHub';
import BentoDashboard from './components/detection/BentoDashboard';
import { sectionSwap } from './theme/motion';

const TeamManifest = lazy(() => import('./components/team/TeamManifest'));

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').trim().replace(/\/+$/, '');

export default function App() {
  const [activeSection, setActiveSection] = useState('detection');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);

  // Detection phase machine: idle → scanning → locking → results
  const [phase, setPhase] = useState('idle');
  const [scanId, setScanId] = useState(0);
  const [scanStatus, setScanStatus] = useState('pending'); // pending | success | error
  const [analysisError, setAnalysisError] = useState('');
  const [apiResults, setApiResults] = useState(null);
  const resultsRef = useRef(null);

  const [apiOnline, setApiOnline] = useState(null);
  const [insightsData, setInsightsData] = useState(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [insightsError, setInsightsError] = useState('');

  const handleImageUpload = useCallback((dataUrl, file) => {
    setImageFile(file);
    setImagePreviewUrl(dataUrl);
    setPhase('idle');
    setApiResults(null);
  }, []);

  const handleRunAnalysis = useCallback(async () => {
    if (!imageFile) return;
    setPhase('scanning');
    setScanStatus('pending');
    setAnalysisError('');
    setApiResults(null);
    setScanId((n) => n + 1);

    try {
      const formData = new FormData();
      formData.append('file', imageFile);

      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setApiOnline(true);
      if (data.success) {
        resultsRef.current = data;
        setApiResults(data);
        setScanStatus('success');
      } else {
        setAnalysisError(`Analysis failed: ${data.error || 'unknown pipeline error'}`);
        setScanStatus('error');
      }
    } catch (err) {
      console.error(err);
      setApiOnline(false);
      setAnalysisError(`No link to the backend at ${API_BASE_URL}. Check VITE_API_BASE_URL and that the server is running.`);
      setScanStatus('error');
    }
  }, [imageFile]);

  const handleScanComplete = useCallback(() => setPhase('locking'), []);
  const handleLockDone = useCallback(() => setPhase('results'), []);
  const handleScanDismiss = useCallback(() => setPhase('idle'), []);
  const handleNewScan = useCallback(() => {
    setPhase('idle');
    setApiResults(null);
  }, []);

  const fetchInsights = useCallback(async (forceReload = false) => {
    if (insightsLoading) return;
    if (!forceReload && insightsData) return;

    setInsightsLoading(true);
    setInsightsError('');

    try {
      const response = await fetch(`${API_BASE_URL}/insights/summary`);
      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Failed to load insights data');
      }
      setApiOnline(true);
      setInsightsData(data);
    } catch (err) {
      console.error(err);
      setApiOnline(false);
      setInsightsError(`Failed to load model insights from ${API_BASE_URL}. Ensure backend is running and ml_results exists.`);
    } finally {
      setInsightsLoading(false);
    }
  }, [insightsData, insightsLoading]);

  useEffect(() => {
    if (activeSection === 'insights') {
      fetchInsights(false);
    }
  }, [activeSection, fetchInsights]);

  return (
    <div className="min-h-screen bg-transparent relative overflow-x-hidden">
      <BackgroundField variant={activeSection === 'team' ? 'crew' : 'scanner'} />
      <TelemetryBar
        activeSection={activeSection}
        onNavigate={setActiveSection}
        apiOnline={apiOnline}
      />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <AnimatePresence mode="wait">
          {activeSection === 'detection' && (
            <Motion.div key="detection" {...sectionSwap} className="space-y-8">
              {phase === 'idle' && (
                <UploadPanel
                  onImageUpload={handleImageUpload}
                  onRunAnalysis={handleRunAnalysis}
                  isLoading={false}
                />
              )}

              {phase === 'scanning' && (
                <ScanSequence
                  key={scanId}
                  image={imagePreviewUrl}
                  status={scanStatus}
                  error={analysisError}
                  onComplete={handleScanComplete}
                  onRetry={handleRunAnalysis}
                  onDismiss={handleScanDismiss}
                />
              )}

              {phase === 'locking' && (
                <TargetLock
                  image={apiResults?.images?.maskOverlay || imagePreviewUrl}
                  potholeCount={apiResults?.potholeCount ?? 0}
                  onDone={handleLockDone}
                />
              )}

              {phase === 'results' && apiResults && (
                <>
                  <div className="max-w-5xl mx-auto flex items-center justify-between gap-3">
                    <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-slate-500">
                      Scan {String(scanId).padStart(3, '0')} · report ready
                    </p>
                    <button
                      type="button"
                      onClick={handleNewScan}
                      className="px-3 py-1.5 rounded-[2px] border border-slate-700 text-slate-400 font-mono text-[10px] uppercase tracking-[0.14em] hover:text-amber-300 hover:border-amber-500/50 transition-colors inline-flex items-center gap-2"
                    >
                      <RotateCcw className="w-3 h-3" /> New scan
                    </button>
                  </div>
                  <BentoDashboard key={scanId} results={apiResults} />
                </>
              )}
            </Motion.div>
          )}

          {activeSection === 'insights' && (
            <Motion.div key="insights" {...sectionSwap} className="space-y-6">
              <InstrumentPanel
                title="Insights Workspace"
                statusLabel="Model diagnostics"
                headerRight={
                  <button
                    type="button"
                    onClick={() => fetchInsights(true)}
                    className="px-2.5 py-1 border border-amber-500/40 bg-amber-500/10 text-amber-300 font-mono text-[10px] uppercase tracking-[0.12em] hover:bg-amber-500/20 transition-all inline-flex items-center gap-1.5 rounded-[2px]"
                  >
                    <RefreshCw className={`w-3 h-3 ${insightsLoading ? 'animate-spin' : ''}`} />
                    Refresh
                  </button>
                }
                bodyClassName="px-4 py-3"
              >
                <p className="text-sm text-slate-400">
                  How well the system works, measured on potholes it never trained on. The current depth network
                  and the retired classifier vote are shown separately, because they were scored in different ways.
                </p>
              </InstrumentPanel>

              <InsightsHub
                data={insightsData}
                loading={insightsLoading}
                error={insightsError}
                apiBase={API_BASE_URL}
              />
            </Motion.div>
          )}

          {activeSection === 'team' && (
            <Motion.div key="team" {...sectionSwap}>
              <Suspense
                fallback={
                  <InstrumentPanel title="Team Roster" statusLabel="Loading" bodyClassName="p-8">
                    <p className="font-mono text-xs text-slate-500 uppercase tracking-widest">
                      Loading channel…
                    </p>
                  </InstrumentPanel>
                }
              >
                <TeamManifest />
              </Suspense>
            </Motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
