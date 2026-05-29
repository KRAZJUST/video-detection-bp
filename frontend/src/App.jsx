import { useState, useEffect, useRef } from 'react';
import './App.css';
import FrameSlideshowModal from './FrameSlideshowModal';
import DirectoryPickerModal from './DirectoryPickerModal';
import ROIDrawer from './ROIDrawer';

const API_BASE = 'http://localhost:8000';

function App() {
  const [videoPath, setVideoPath] = useState('');
  const [videoInfo, setVideoInfo] = useState(null);
  const [tracker, setTracker] = useState('yolo');
  const [interval, setInterval] = useState(30);
  const [segmentation, setSegmentation] = useState(false);
  const [query, setQuery] = useState('');
  const [theme, setTheme] = useState('light');
  const [outputDir, setOutputDir] = useState('./outputs/');
  const [taskId, setTaskId] = useState(null);

  const [xclipBatchCount, setXclipBatchCount] = useState(5);
  const [siglipFrameCount, setSiglipFrameCount] = useState(40);
  const [deduplicate, setDeduplicate] = useState(true);
  const [showQuerySettings, setShowQuerySettings] = useState(false);

  const [roi, setRoi] = useState(null);

  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');

  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState([]);

  const [selectedFrame, setSelectedFrame] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isDirPickerOpen, setIsDirPickerOpen] = useState(false);

  const fileInputRef = useRef(null);

  // Apply theme to document root
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  const fetchVideoInfo = async (path) => {
    try {
      const res = await fetch(`${API_BASE}/api/video-info?path=${encodeURIComponent(path)}`);
      if (res.ok) {
        const data = await res.json();
        setVideoInfo(data);
      }
    } catch (e) {
      console.error('Error fetching video info:', e);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Clear previous search results and state
    setSearchResults([]);
    setProgress(0);
    setTaskId(null);

    // If we already have a video uploaded, delete it to save space
    if (videoPath) {
      fetch(`${API_BASE}/api/video?path=${encodeURIComponent(videoPath)}`, { method: 'DELETE' })
        .catch(err => console.error('Error deleting previous video:', err));
    }

    const formData = new FormData();
    formData.append('file', file);

    setStatusMessage('Uploading video...');
    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const errData = await res.text();
        throw new Error(`Server returned ${res.status}: ${errData}`);
      }
      const data = await res.json();
      setVideoPath(data.path);
      setStatusMessage('Video uploaded successfully.');
      fetchVideoInfo(data.path);
    } catch (err) {
      console.error(err);
      setStatusMessage(`Upload failed: ${err.message}`);
    }
  };

  const handleProcessClick = async () => {
    if (!videoPath) {
      alert("Please upload a video first.");
      return;
    }

    setIsProcessing(true);
    setProgress(0);
    setStatusMessage('Starting processing...');

    const formData = new FormData();
    formData.append('video_path', videoPath);
    formData.append('interval', interval);
    formData.append('tracker', tracker);
    formData.append('segmentation', segmentation);
    formData.append('output_dir', outputDir);

    try {
      const res = await fetch(`${API_BASE}/api/process`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setTaskId(data.task_id);

      // Start SSE to listen for progress
      const eventSource = new EventSource(`${API_BASE}/api/process/status/${data.task_id}`);

      eventSource.onmessage = (event) => {
        const taskData = JSON.parse(event.data);
        setProgress(taskData.progress);
        setStatusMessage(taskData.message);

        if (taskData.status === 'completed' || taskData.status === 'error') {
          eventSource.close();
          setIsProcessing(false);
          if (taskData.status === 'completed') {
            // Refetch video info if needed, though we already have it
          }
        }
      };

      eventSource.onerror = () => {
        eventSource.close();
        setIsProcessing(false);
        setStatusMessage('Error tracking progress.');
      };

    } catch (err) {
      console.error(err);
      setIsProcessing(false);
      setStatusMessage('Processing failed to start.');
    }
  };

  const handleCancelProcessing = async () => {
    if (!taskId) return;
    try {
      await fetch(`${API_BASE}/api/process/cancel/${taskId}`, { method: 'POST' });
      setStatusMessage('Cancellation requested. Waiting for process to stop...');
    } catch (err) {
      console.error(err);
      setStatusMessage('Error cancelling process.');
    }
  };

  const handleSearchClick = async () => {
    if (!videoPath || !query) return;

    setIsSearching(true);
    setSearchResults([]);

    const formData = new FormData();
    formData.append('query', query);
    formData.append('video_path', videoPath);
    formData.append('tracker', tracker);
    formData.append('segmentation', segmentation);
    formData.append('deduplicate', deduplicate);
    formData.append('xclip_batch_count', xclipBatchCount);
    formData.append('siglip_frame_count', siglipFrameCount);
    formData.append('output_dir', outputDir);
    if (roi) {
      formData.append('roi', roi.join(','));
    }

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (data.results) {
        const normalizedOutputDir = outputDir ? outputDir.replace(/\/+$/, '') : './outputs';

        // Transform the results dictionary to an array for rendering
        const resultsArray = Object.entries(data.results).map(([frameKey, detections]) => {
          let filename = frameKey.split('/').pop().split('\\').pop();

          // If the key is just a number (from DetectionParser), format it correctly
          if (/^\d+$/.test(filename)) {
            filename = `frame_${filename.padStart(6, '0')}.jpg`;
          }

          const absoluteFramePath = `${normalizedOutputDir}/found_frames/${filename}`;

          return {
            framePath: absoluteFramePath,
            filename,
            detections
          };
        });
        setSearchResults(resultsArray);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsSearching(false);
    }
  };

  // Helper to format duration
  const formatDuration = (seconds) => {
    if (!seconds) return '00:00:00';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="app-container">
      {/* Sidebar Controls */}
      <aside className="sidebar">
        <div style={{ padding: '2rem 1.5rem', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 className="app-title">Video Analysis</h1>
            <p className="text-secondary" style={{ fontSize: '0.85rem' }}>Advanced semantic search in videos</p>
          </div>
          <button className="theme-toggle" onClick={toggleTheme} title="Toggle Theme">
            {theme === 'light' ? '🌙' : '☀️'}
          </button>
        </div>

        <div style={{ padding: '1.5rem', flex: 1, overflowY: 'auto' }}>
          {/* Input/Output Group */}
          <div style={{ marginBottom: '2rem' }}>
            <h3 style={{ fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)', marginBottom: '1rem' }}>Data Source</h3>

            <div className="input-group">
              <label className="input-label">Video Path</label>
              <input
                type="text"
                className="input-field"
                placeholder="Select or enter video path..."
                value={videoPath}
                readOnly
              />
              <input
                type="file"
                accept="video/*"
                ref={fileInputRef}
                style={{ display: 'none' }}
                onChange={handleFileUpload}
              />
              <button
                className="btn btn-secondary"
                style={{ marginTop: '0.25rem' }}
                onClick={() => fileInputRef.current.click()}
              >
                Browse File
              </button>
            </div>

            <div className="input-group" style={{ marginTop: '1rem' }}>
              <label className="input-label">Output Directory</label>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <input
                  type="text"
                  className="input-field"
                  placeholder="./outputs/"
                  value={outputDir}
                  onChange={(e) => setOutputDir(e.target.value)}
                />
                <button
                  className="btn btn-secondary"
                  onClick={() => setIsDirPickerOpen(true)}
                  title="Browse for output directory"
                >
                  📁
                </button>
              </div>
            </div>
          </div>

          {/* Processing Settings Group */}
          <div style={{ marginBottom: '2rem' }}>
            <h3 style={{ fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)', marginBottom: '1rem' }}>Processing</h3>

            <div className="input-group">
              <label className="input-label">Detection Engine</label>
              <select
                className="input-field"
                value={tracker}
                onChange={(e) => setTracker(e.target.value)}
                style={{ appearance: 'none', backgroundColor: 'var(--bg-surface-elevated)' }}
              >
                <option value="yolo">YOLOv11</option>
                <option value="bytetrack">ByteTrack</option>
                <option value="xclip-32">X-CLIP</option>
                <option value="siglip">SigLIP</option>
              </select>
            </div>

            <div className="input-group">
              <label className="input-label">Frame Interval</label>
              <input
                type="number"
                className="input-field"
                value={interval}
                onChange={(e) => setInterval(Number(e.target.value))}
                min="1"
                max="200"
              />
            </div>
          </div>



          {/* Action Button */}
          <div style={{ marginTop: 'auto', paddingTop: '1rem' }}>
            <button
              className="btn btn-primary"
              style={{ width: '100%', padding: '1rem' }}
              onClick={handleProcessClick}
              disabled={isProcessing || !videoPath}
            >
              {isProcessing ? 'Processing...' : 'Analyze Video'}
            </button>

            {isProcessing && (
              <button
                className="btn"
                style={{ width: '100%', padding: '1rem', marginTop: '0.5rem', backgroundColor: '#ff4d4f', color: 'white', border: 'none' }}
                onClick={handleCancelProcessing}
              >
                Cancel Processing
              </button>
            )}

            {statusMessage && (
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.5rem', textAlign: 'center' }}>
                {statusMessage}
              </p>
            )}

            {isProcessing && (
              <div className="progress-container">
                <div className="progress-bar" style={{ width: `${progress}%` }}></div>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {/* Top Header / Query Builder */}
        <header className="glass-surface" style={{ margin: '1.5rem', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-end' }}>
            <div className="input-group" style={{ flex: 1, marginBottom: 0 }}>
              <label className="input-label" style={{ display: 'flex', justifyContent: 'space-between' }}>
                Semantic Search Query
              </label>
              <input
                type="text"
                className="input-field"
                placeholder="e.g., 'Person wearing a red jacket running'"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                style={{ fontSize: '1.1rem', padding: '1rem' }}
                onKeyDown={(e) => e.key === 'Enter' && handleSearchClick()}
              />
            </div>
            <button
              className="btn btn-secondary"
              style={{ padding: '1rem', height: 'fit-content' }}
              onClick={() => setShowQuerySettings(!showQuerySettings)}
              title="Query Settings"
            >
              ⚙️
            </button>
            <button
              className="btn btn-primary"
              style={{ padding: '1rem 2rem', height: 'fit-content' }}
              onClick={handleSearchClick}
              disabled={isSearching || !videoPath || !query}
            >
              {isSearching ? 'Searching...' : 'Search'}
            </button>
          </div>

          {showQuerySettings && (
            <div className="animate-fade-in" style={{ padding: '1rem', background: 'var(--bg-surface-elevated)', borderRadius: '8px', border: '1px solid var(--border-color)', display: 'flex', gap: '2rem', flexWrap: 'wrap', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <label className="input-label" style={{ marginBottom: 0 }}>XCLIP Batch Count:</label>
                <input
                  type="number"
                  className="input-field"
                  value={xclipBatchCount}
                  onChange={(e) => setXclipBatchCount(Number(e.target.value))}
                  min="1"
                  style={{ width: '80px', padding: '0.5rem' }}
                />
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <label className="input-label" style={{ marginBottom: 0 }}>SigLIP Frame Count:</label>
                <input
                  type="number"
                  className="input-field"
                  value={siglipFrameCount}
                  onChange={(e) => setSiglipFrameCount(Number(e.target.value))}
                  min="1"
                  style={{ width: '80px', padding: '0.5rem' }}
                />
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <input
                  type="checkbox"
                  id="deduplicate-header"
                  checked={deduplicate}
                  onChange={(e) => setDeduplicate(e.target.checked)}
                  style={{ accentColor: 'var(--primary-accent)', width: '16px', height: '16px' }}
                />
                <label htmlFor="deduplicate-header" style={{ fontSize: '0.9rem', color: 'var(--text-primary)' }}>Deduplicate Frames</label>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <input
                  type="checkbox"
                  id="segmentation-header"
                  checked={segmentation}
                  onChange={(e) => setSegmentation(e.target.checked)}
                  style={{ accentColor: 'var(--primary-accent)', width: '16px', height: '16px' }}
                />
                <label htmlFor="segmentation-header" style={{ fontSize: '0.9rem', color: 'var(--text-primary)' }}>Use Segmentation</label>
              </div>
            </div>
          )}
        </header>

        {/* Video Info & Stats Bar */}
        <div className="glass-surface" style={{ margin: '0 1.5rem 1.5rem 1.5rem', padding: '1rem 1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', gap: '1.5rem', color: 'var(--text-secondary)', fontSize: '0.85rem', flexWrap: 'wrap' }}>
            <span><strong style={{ color: 'var(--text-primary)' }}>Duration:</strong> {videoInfo ? formatDuration(videoInfo.duration) : '-'}</span>
            <span><strong style={{ color: 'var(--text-primary)' }}>Resolution:</strong> {videoInfo ? `${videoInfo.width}x${videoInfo.height}` : '-'}</span>
            <span><strong style={{ color: 'var(--text-primary)' }}>FPS:</strong> {videoInfo ? videoInfo.fps : '-'}</span>
            <span><strong style={{ color: 'var(--text-primary)' }}>Total Frames:</strong> {videoInfo ? videoInfo.frame_count?.toLocaleString() : '-'}</span>
            <span><strong style={{ color: 'var(--text-primary)' }}>Codec:</strong> {videoInfo ? videoInfo.codec : '-'}</span>
            <span><strong style={{ color: 'var(--text-primary)' }}>Bitrate:</strong> {videoInfo ? (parseInt(videoInfo.bitrate) / 1000000).toFixed(2) + ' Mbps' : '-'}</span>
            <span><strong style={{ color: 'var(--text-primary)' }}>File Size:</strong> {videoInfo ? videoInfo.size : '-'}</span>
          </div>
          <div style={{ fontSize: '0.9rem', color: 'var(--success)', fontWeight: 500 }}>
            Found {searchResults.length} matching frames
          </div>
        </div>

        {/* Video Preview & ROI Selection */}
        {videoPath && videoInfo && (
          <div className="glass-surface" style={{ margin: '0 1.5rem 1.5rem 1.5rem', padding: '1.5rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <h3 style={{ fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)', marginBottom: '1rem', alignSelf: 'flex-start' }}>Video Preview & ROI Selection</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem', alignSelf: 'flex-start' }}>
              Hold <strong>Shift</strong> and drag on the video to draw a Region of Interest (ROI). Only detections inside this region will be matched. ROI filtering works only
              with YOLO and ByteTrack.
            </p>
            <div style={{ position: 'relative', display: 'inline-block' }}>
              <video
                id="preview-video"
                src={`${API_BASE}/api/file?path=${encodeURIComponent(videoPath)}`}
                controls
                style={{ maxHeight: '500px', maxWidth: '100%', display: 'block', borderRadius: '8px', border: '1px solid var(--border-color)' }}
              />
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, zIndex: 10, pointerEvents: 'none' }}>
                <ROIDrawer
                  width={videoInfo.width}
                  height={videoInfo.height}
                  onROIChange={(rect) => {
                    if (!rect) {
                      setRoi(null);
                      return;
                    }
                    setRoi(rect);
                  }}
                />
              </div>
            </div>
            {roi && (
              <p style={{ marginTop: '1rem', fontSize: '0.85rem', color: 'var(--primary-accent)', fontWeight: 500 }}>
                Active ROI selected: ({roi.join(', ')})
              </p>
            )}
          </div>
        )}

        {/* Results Gallery */}
        <div className="results-grid">
          {searchResults.map((item, idx) => (
            <div
              key={idx}
              className="glass-surface animate-fade-in"
              style={{ overflow: 'hidden', animationDelay: `${(idx % 10) * 0.05}s`, cursor: 'pointer', transition: 'transform 0.2s', '&:hover': { transform: 'scale(1.02)' } }}
              onClick={() => {
                setSelectedFrame(item);
                setIsModalOpen(true);
              }}
            >
              <div style={{ position: 'relative', width: '100%', paddingBottom: '56.25%', backgroundColor: 'var(--bg-main)' }}>
                <img
                  src={`${API_BASE}/api/file?path=${encodeURIComponent(item.framePath)}`}
                  alt="Detection result"
                  onError={(e) => {
                    // Fallback if the image doesn't exist or isn't in found_frames (e.g., XClip generic results)
                    e.target.onerror = null;
                    e.target.src = '/dummy.png';
                  }}
                  style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'cover' }}
                />
              </div>
              <div style={{ padding: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.9rem', fontWeight: 500 }} title={item.filename}>
                    {item.filename.length > 20 ? item.filename.substring(0, 20) + '...' : item.filename}
                  </span>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  {Array.isArray(item.detections) && item.detections.map((det, dIdx) => (
                    <span key={dIdx} className="result-tag" title={`Confidence: ${det.confidence}`}>
                      {det.class_name || det.dominant_color || 'detection'}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </main>

      <FrameSlideshowModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        frameItem={selectedFrame}
        tracker={tracker}
        API_BASE={API_BASE}
        fps={videoInfo ? videoInfo.fps : null}
        interval={interval}
        videoPath={videoPath}
        outputDir={outputDir}
      />

      <DirectoryPickerModal
        isOpen={isDirPickerOpen}
        onClose={() => setIsDirPickerOpen(false)}
        onSelect={(path) => setOutputDir(path)}
        initialPath={outputDir}
        API_BASE={API_BASE}
      />
    </div>
  );
}

export default App;
