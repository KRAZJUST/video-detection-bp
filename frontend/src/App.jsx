import { useState, useEffect, useRef, useMemo } from 'react';
import './App.css';
import FrameSlideshowModal from './FrameSlideshowModal';
import DirectoryPickerModal from './DirectoryPickerModal';
import ROIDrawer from './ROIDrawer';
import { useToast } from './ToastProvider';

const API_BASE = 'http://localhost:8000';

const STAT_COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#8b5cf6', '#14b8a6', '#f97316', '#06b6d4', '#84cc16'];

function App() {
  const { addToast } = useToast();

  const [videoPath, setVideoPath] = useState('');
  const [videoInfo, setVideoInfo] = useState(null);
  const [tracker, setTracker] = useState('yolo');
  const [interval, setInterval] = useState(30);
  const [segmentation, setSegmentation] = useState(false);
  const [query, setQuery] = useState('');
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('neurovision-theme') || 'light'; }
    catch { return 'light'; }
  });
  const [outputDir, setOutputDir] = useState('./outputs/');
  const [taskId, setTaskId] = useState(null);

  const [xclipBatchCount, setXclipBatchCount] = useState(5);
  const [siglipFrameCount, setSiglipFrameCount] = useState(40);
  const [deduplicate, setDeduplicate] = useState(true);
  const [showQuerySettings, setShowQuerySettings] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0);

  const [roi, setRoi] = useState(null);

  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');

  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [searchMetadata, setSearchMetadata] = useState(null);
  const [hasSearched, setHasSearched] = useState(false);

  const [selectedFrame, setSelectedFrame] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isDirPickerOpen, setIsDirPickerOpen] = useState(false);

  // New feature states
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isSidebarDragOver, setIsSidebarDragOver] = useState(false);
  const [searchHistory, setSearchHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem('neurovision-search-history') || '[]'); }
    catch { return []; }
  });

  // Processing log
  const [processingLog, setProcessingLog] = useState([]);
  const [showProcessingLog, setShowProcessingLog] = useState(false);
  const logEndRef = useRef(null);

  const fileInputRef = useRef(null);
  const searchInputRef = useRef(null);

  // Apply theme to document root and persist
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('neurovision-theme', theme); } catch {}
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ctrl+K or Cmd+K to focus search
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        searchInputRef.current?.focus();
        return;
      }
      // Escape to close modals
      if (e.key === 'Escape') {
        if (isModalOpen) setIsModalOpen(false);
        else if (isDirPickerOpen) setIsDirPickerOpen(false);
        return;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isModalOpen, isDirPickerOpen]);

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

  const uploadFile = async (file) => {
    if (!file) return;

    setSearchResults([]);
    setProgress(0);
    setTaskId(null);

    if (videoPath) {
      fetch(`${API_BASE}/api/video?path=${encodeURIComponent(videoPath)}`, { method: 'DELETE' })
        .catch(err => console.error('Error deleting previous video:', err));
    }

    const formData = new FormData();
    formData.append('file', file);

    addToast('Uploading video...', 'info');
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
      addToast('Video uploaded successfully!', 'success');
      setSidebarCollapsed(false);
      fetchVideoInfo(data.path);
    } catch (err) {
      console.error(err);
      setStatusMessage(`Upload failed: ${err.message}`);
      addToast(`Upload failed: ${err.message}`, 'error');
    }
  };

  const handleFileUpload = (e) => {
    uploadFile(e.target.files[0]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    setIsSidebarDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      uploadFile(file);
    } else {
      addToast('Please drop a video file.', 'error');
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
    setProcessingLog([]);
    setShowProcessingLog(true);
    addToast('Starting processing...', 'info');

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

        // Append to processing log
        if (taskData.message) {
          setProcessingLog(prev => [...prev, { time: new Date().toLocaleTimeString(), message: taskData.message, progress: taskData.progress }]);
        }

        if (taskData.status === 'completed' || taskData.status === 'error') {
          eventSource.close();
          setIsProcessing(false);
          if (taskData.status === 'completed') {
            addToast('Processing completed successfully!', 'success');
            // Refetch video info if needed, though we already have it
          } else {
            addToast(`Processing error: ${taskData.message}`, 'error');
          }
        }
      };

      eventSource.onerror = () => {
        eventSource.close();
        setIsProcessing(false);
        setStatusMessage('Error tracking progress.');
        addToast('Error tracking progress.', 'error');
      };

    } catch (err) {
      console.error(err);
      setIsProcessing(false);
      setStatusMessage('Processing failed to start.');
      addToast('Processing failed to start.', 'error');
    }
  };

  const handleCancelProcessing = async () => {
    if (!taskId) return;
    try {
      await fetch(`${API_BASE}/api/process/cancel/${taskId}`, { method: 'POST' });
      setStatusMessage('Cancellation requested. Waiting for process to stop...');
      addToast('Cancellation requested.', 'info');
    } catch (err) {
      console.error(err);
      setStatusMessage('Error cancelling process.');
      addToast('Error cancelling process.', 'error');
    }
  };

  const handleSearchClick = async () => {
    if (!videoPath || !query) return;

    setIsSearching(true);
    setSearchResults([]);
    setHasSearched(true);

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
        setSearchMetadata(data.metadata || null);
        addToast(`Found ${resultsArray.length} matching frames!`, 'success');

        // Update Search History
        setSearchHistory(prev => {
          const filtered = prev.filter(q => q.toLowerCase() !== query.toLowerCase());
          const newHistory = [query, ...filtered].slice(0, 10);
          localStorage.setItem('neurovision-search-history', JSON.stringify(newHistory));
          return newHistory;
        });
      }
    } catch (err) {
      console.error(err);
      addToast('Search failed.', 'error');
    } finally {
      setIsSearching(false);
    }
  };

  const removeHistoryItem = (e, itemToRemove) => {
    e.stopPropagation();
    setSearchHistory(prev => {
      const newHistory = prev.filter(item => item !== itemToRemove);
      localStorage.setItem('neurovision-search-history', JSON.stringify(newHistory));
      return newHistory;
    });
  };

  const clearResults = () => {
    setSearchResults([]);
    setSearchMetadata(null);
    setHasSearched(false);
    setQuery('');
  };

  // Auto-scroll processing log
  useEffect(() => {
    if (logEndRef.current && showProcessingLog) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [processingLog, showProcessingLog]);

  // Helper to format duration
  const formatDuration = (seconds) => {
    if (!seconds) return '00:00:00';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  // Filter results by confidence threshold
  const filteredResults = useMemo(() => {
    if (confidenceThreshold <= 0) return searchResults;
    return searchResults.filter(item => {
      if (!Array.isArray(item.detections)) return true;
      return item.detections.some(det => (det.confidence || 0) >= confidenceThreshold / 100);
    });
  }, [searchResults, confidenceThreshold]);

  const detectionStats = useMemo(() => {
    if (!filteredResults.length) return null;
    let total = 0;
    const counts = {};

    // Extract filter objects and colors from metadata if available
    const filterObjects = searchMetadata?.filter_objects || [];
    const filterColors = searchMetadata?.filter_colors || [];

    filteredResults.forEach(res => {
      if (res.detections) {
        res.detections.forEach(det => {
          const className = det.class_name || 'unknown';
          const color = det.dominant_color;

          // If filters are active, only count detections that match the search
          if (filterObjects.length > 0 && !filterObjects.includes(className)) {
            return;
          }
          if (filterColors.length > 0 && color && !filterColors.includes(color)) {
            return;
          }
          // Confidence threshold filter
          if (confidenceThreshold > 0 && (det.confidence || 0) < confidenceThreshold / 100) {
            return;
          }

          const key = className || color || 'unknown';
          counts[key] = (counts[key] || 0) + 1;
          total++;
        });
      }
    });
    const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    return { total, sorted };
  }, [filteredResults, searchMetadata, confidenceThreshold]);

  return (
    <div className="app-container" onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }} onDragLeave={(e) => { if (e.currentTarget === e.target) setIsDragOver(false); }} onDrop={handleDrop}>
      {/* Sidebar Controls */}
      <aside className={`sidebar ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-content">
            <h1 className="app-title">Video Analysis</h1>
            <p className="text-secondary" style={{ fontSize: '0.85rem' }}>Advanced semantic search in videos</p>
          </div>
          <div className="sidebar-header-actions">
            <button className="theme-toggle" onClick={toggleTheme} title="Toggle Theme" style={{ width: '32px', height: '32px', padding: 0 }}>
              {theme === 'light' ? (
                <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="currentColor"><path d="M600-640 480-760l120-120 120 120-120 120Zm200 120-80-80 80-80 80 80-80 80ZM483-80q-84 0-157.5-32t-128-86.5Q143-253 111-326.5T79-484q0-146 93-257.5T409-880q-18 99 11 193.5T520-521q71 71 165.5 100T879-410q-26 144-138 237T483-80Zm0-80q88 0 163-44t118-121q-86-8-163-43.5T463-465q-61-61-97-138t-43-163q-77 43-120.5 118.5T159-484q0 135 94.5 229.5T483-160Zm-20-305Z" /></svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="currentColor"><path d="M440-760v-160h80v160h-80Zm266 110-55-55 112-115 56 57-113 113Zm54 210v-80h160v80H760ZM440-40v-160h80v160h-80ZM254-652 140-763l57-56 113 113-56 54Zm508 512L651-255l54-54 114 110-57 59ZM40-440v-80h160v80H40Zm157 300-56-57 112-112 29 27 29 28-114 114Zm113-170q-70-70-70-170t70-170q70-70 170-70t170 70q70 70 70 170t-70 170q-70 70-170 70t-170-70Zm283-57q47-47 47-113t-47-113q-47-47-113-47t-113 47q-47 47-47 113t47 113q47 47 113 47t113-47ZM480-480Z" /></svg>
              )}
            </button>
            <button className="sidebar-toggle" onClick={() => setSidebarCollapsed(!sidebarCollapsed)} title="Toggle Sidebar">
              <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="currentColor"><path d="M400-240 160-480l240-240 56 58-142 142h486v80H314l142 142-56 58Z" /></svg>
            </button>
          </div>
        </div>

        <div className="sidebar-content" style={{ padding: '1.5rem', flex: 1, overflowY: 'auto' }}>
          {/* Input/Output Group */}
          <div style={{ marginBottom: '2rem' }}>
            <h3 style={{ fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)', marginBottom: '1rem' }}>Data Source</h3>

            <div className="input-group">
              <label className="input-label">Video Source</label>

              <div
                className={`small-drop-zone ${isSidebarDragOver ? 'drag-over' : ''} ${videoPath ? 'has-file' : ''}`}
                onClick={() => fileInputRef.current.click()}
                onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); setIsSidebarDragOver(true); }}
                onDragLeave={(e) => { if (e.currentTarget === e.target) setIsSidebarDragOver(false); }}
                onDrop={handleDrop}
              >
                <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="currentColor">
                  <path d="M440-200h80v-167l64 64 56-57-160-160-160 160 57 56 63-63v167ZM240-80q-33 0-56.5-23.5T160-160v-640q0-33 23.5-56.5T240-880h320l240 240v480q0 33-23.5 56.5T720-80H240Zm280-520v-200H240v640h480v-440H520ZM240-800v200-200 640-640Z" />
                </svg>
                <div className="small-drop-text">
                  {videoPath ? videoPath.split('/').pop().split('\\').pop() : 'Click or drop video here'}
                </div>
              </div>

              <input
                type="file"
                accept="video/*"
                style={{ display: 'none' }}
                ref={fileInputRef}
                onChange={handleFileUpload}
              />
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
                  <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="currentColor"><path d="M160-160q-33 0-56.5-23.5T80-240v-480q0-33 23.5-56.5T160-800h240l80 80h320q33 0 56.5 23.5T880-640v242q-18-14-38-23t-42-19v-200H447l-80-80H160v480h120v80H160ZM640-40q-91 0-168-48T360-220q35-84 112-132t168-48q91 0 168 48t112 132q-35 84-112 132T640-40Zm107.5-106q50.5-26 82.5-74-32-48-82.5-74T640-320q-57 0-107.5 26T450-220q32 48 82.5 74T640-120q57 0 107.5-26Zm-150-31.5Q580-195 580-220t17.5-42.5Q615-280 640-280t42.5 17.5Q700-245 700-220t-17.5 42.5Q665-160 640-160t-42.5-17.5ZM160-240v-480 277-37 240Z" /></svg>
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
          <div style={{ marginTop: 'auto', paddingTop: '1rem', display: 'flex', flexDirection: 'column' }}>
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
        {!videoPath ? (
          <div className="empty-state">
            <div
              className={`drop-zone ${isDragOver ? 'drag-over' : ''}`}
              onClick={() => fileInputRef.current.click()}
            >
              <svg className="drop-zone-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960" fill="currentColor">
                <path d="M440-200h80v-167l64 64 56-57-160-160-160 160 57 56 63-63v167ZM240-80q-33 0-56.5-23.5T160-160v-640q0-33 23.5-56.5T240-880h320l240 240v480q0 33-23.5 56.5T720-80H240Zm280-520v-200H240v640h480v-440H520ZM240-800v200-200 640-640Z" />
              </svg>
              <div className="drop-zone-title">Drop your video here</div>
              <div className="drop-zone-subtitle">or click to browse your files</div>
            </div>
            {/* Optional hero image below */}
            <div style={{ marginTop: '3rem', opacity: 0.5, pointerEvents: 'none', maxWidth: '300px' }}>
              <img src="/hero.png" alt="" style={{ width: '100%', height: 'auto', filter: theme === 'dark' ? 'invert(1) hue-rotate(180deg)' : 'none' }} />
            </div>
          </div>
        ) : (
          <>
            {/* Top Header / Query Builder */}
            <header className="glass-surface" style={{ margin: '1.5rem', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
                <div className="input-group" style={{ flex: 1, marginBottom: 0 }}>
                  <label className="input-label" style={{ display: 'flex', justifyContent: 'space-between' }}>
                    Semantic Search Query
                  </label>
                  <input
                    type="text"
                    className="input-field"
                    ref={searchInputRef}
                    placeholder={
                      tracker === 'yolo' ? "e.g., 'yellow truck and white car'" :
                        tracker === 'bytetrack' ? "e.g., 'white car going north'" :
                          tracker === 'xclip-32' ? "e.g., 'person wearing a red jacket running'" :
                            tracker === 'siglip' ? "e.g., 'person wearing a red jacket'" :
                              "e.g., 'yellow truck and white car'"
                    }
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    style={{ fontSize: '1.1rem', padding: '1rem' }}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearchClick()}
                  />
                  {searchHistory.length > 0 && (
                    <div className="search-history">
                      {searchHistory.map((item, idx) => (
                        <div key={idx} className="search-history-chip" onClick={() => { setQuery(item); }}>
                          <svg xmlns="http://www.w3.org/2000/svg" height="14" viewBox="0 -960 960 960" width="14" fill="currentColor"><path d="m382-80-43-43 297-297H120v-60h516L339-777l43-43 378 378L382-80Z" /></svg>
                          {item}
                          <span className="chip-remove" onClick={(e) => removeHistoryItem(e, item)}>×</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  className="btn btn-secondary"
                  style={{ padding: '1rem', height: 'fit-content', backgroundColor: showQuerySettings ? 'var(--text-primary)' : '', color: showQuerySettings ? 'var(--bg-surface)' : '', marginTop: '1.6rem' }}
                  onClick={() => setShowQuerySettings(!showQuerySettings)}
                  title="Query Settings"
                >
                  <div style={{
                    width: '1.2rem',
                    height: '1.2rem',
                    backgroundColor: 'currentColor',
                    mask: 'url(/settings-icon.png) no-repeat center / contain',
                    WebkitMask: 'url(/settings-icon.png) no-repeat center / contain'
                  }} />
                </button>
                <button
                  className="btn btn-primary"
                  style={{ padding: '1rem 2rem', height: 'fit-content', marginTop: '1.6rem' }}
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

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', minWidth: '220px' }}>
                    <label className="input-label" style={{ marginBottom: 0, whiteSpace: 'nowrap' }}>Min Confidence:</label>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={confidenceThreshold}
                      onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                      style={{ flex: 1, accentColor: 'var(--primary-accent)' }}
                    />
                    <span style={{ fontSize: '0.85rem', color: 'var(--text-primary)', fontWeight: 500, minWidth: '35px' }}>{confidenceThreshold}%</span>
                  </div>
                </div>
              )}
            </header>

            {/* Video Info & Stats Bar */}
            <div className="glass-surface" style={{ margin: '0 1.5rem 1.5rem 1.5rem', padding: '1rem 1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                <div style={{ display: 'flex', gap: '1.5rem', color: 'var(--text-secondary)', fontSize: '0.85rem', flexWrap: 'wrap' }}>
                  <span><strong style={{ color: 'var(--text-primary)' }}>Duration:</strong> {videoInfo ? formatDuration(videoInfo.duration) : '-'}</span>
                  <span><strong style={{ color: 'var(--text-primary)' }}>Resolution:</strong> {videoInfo ? `${videoInfo.width}x${videoInfo.height}` : '-'}</span>
                  <span><strong style={{ color: 'var(--text-primary)' }}>FPS:</strong> {videoInfo ? videoInfo.fps : '-'}</span>
                  <span><strong style={{ color: 'var(--text-primary)' }}>Total Frames:</strong> {videoInfo ? videoInfo.frame_count?.toLocaleString() : '-'}</span>
                  <span><strong style={{ color: 'var(--text-primary)' }}>Codec:</strong> {videoInfo ? videoInfo.codec : '-'}</span>
                  <span><strong style={{ color: 'var(--text-primary)' }}>Bitrate:</strong> {videoInfo ? (parseInt(videoInfo.bitrate) / 1000000).toFixed(2) + ' Mbps' : '-'}</span>
                  <span><strong style={{ color: 'var(--text-primary)' }}>File Size:</strong> {videoInfo ? videoInfo.size : '-'}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  {filteredResults.length > 0 && (
                    <span className="results-badge">
                      {filteredResults.length} result{filteredResults.length !== 1 ? 's' : ''}
                      {confidenceThreshold > 0 && ` (≥${confidenceThreshold}%)`}
                    </span>
                  )}
                  {hasSearched && (
                    <button
                      className="btn btn-secondary"
                      style={{ padding: '0.4rem 0.75rem', fontSize: '0.8rem' }}
                      onClick={clearResults}
                    >
                      Clear Results
                    </button>
                  )}
                </div>
              </div>

              {/* Detection Statistics Panel */}
              {detectionStats && detectionStats.total > 0 && (
                <div className="stats-panel" style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
                  <div className="stats-bar-container">
                    <div className="stats-bar">
                      {detectionStats.sorted.map(([name, count], idx) => (
                        <div
                          key={name}
                          className="stats-segment"
                          style={{ width: `${(count / detectionStats.total) * 100}%`, backgroundColor: STAT_COLORS[idx % STAT_COLORS.length] }}
                          title={`${name}: ${count} (${Math.round((count / detectionStats.total) * 100)}%)`}
                        />
                      ))}
                    </div>
                    <div className="stats-legend">
                      {detectionStats.sorted.slice(0, 6).map(([name, count], idx) => (
                        <div key={name} className="stats-legend-item">
                          <div className="stats-legend-dot" style={{ backgroundColor: STAT_COLORS[idx % STAT_COLORS.length] }} />
                          <span>{name} <span className="stats-legend-count">{count}</span></span>
                        </div>
                      ))}
                      {detectionStats.sorted.length > 6 && (
                        <div className="stats-legend-item">
                          <span style={{ opacity: 0.6 }}>+ {detectionStats.sorted.length - 6} more classes</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Video Preview & ROI Selection */}
            {videoInfo && (
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
                  <p style={{ marginTop: '1rem', fontSize: '0.85rem', color: 'var(--text-primary)', fontWeight: 500 }}>
                    Active ROI selected: ({roi.join(', ')})
                  </p>
                )}
              </div>
            )}

            {/* Results Gallery */}
            <div className="results-grid">
              {isSearching ? (
                // Skeleton loading state
                Array.from({ length: 6 }).map((_, idx) => (
                  <div key={`skeleton-${idx}`} className="skeleton-card">
                    <div className="skeleton-image"></div>
                    <div className="skeleton-text">
                      <div className="skeleton-line skeleton-line-short"></div>
                      <div className="skeleton-line-tags">
                        <div className="skeleton-tag"></div>
                        <div className="skeleton-tag" style={{ width: '3rem' }}></div>
                      </div>
                    </div>
                  </div>
                ))
              ) : filteredResults.length === 0 && hasSearched ? (
                // No results empty state
                <div className="no-results-state" style={{ gridColumn: '1 / -1' }}>
                  <svg xmlns="http://www.w3.org/2000/svg" height="64px" viewBox="0 -960 960 960" width="64px" fill="currentColor" style={{ opacity: 0.3 }}>
                    <path d="M784-120 532-372q-30 24-69 38t-83 14q-109 0-184.5-75.5T120-580q0-109 75.5-184.5T380-840q109 0 184.5 75.5T640-580q0 44-14 83t-38 69l252 252-56 56ZM380-400q75 0 127.5-52.5T560-580q0-75-52.5-127.5T380-760q-75 0-127.5 52.5T200-580q0 75 52.5 127.5T380-400Z"/>
                  </svg>
                  <h3 style={{ color: 'var(--text-primary)', marginTop: '1rem' }}>No results found</h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', maxWidth: '400px', textAlign: 'center' }}>
                    Try adjusting your search query, lowering the confidence threshold, or using a different detection engine.
                  </p>
                  {confidenceThreshold > 0 && (
                    <button
                      className="btn btn-secondary"
                      style={{ marginTop: '1rem' }}
                      onClick={() => setConfidenceThreshold(0)}
                    >
                      Reset Confidence Filter
                    </button>
                  )}
                </div>
              ) : (
                filteredResults.map((item, idx) => {
                  const filterObjects = searchMetadata?.filter_objects || [];
                  const filterColors = searchMetadata?.filter_colors || [];

                  // Filter the detections to only those relevant to the query for UI display
                  const relevantDetections = Array.isArray(item.detections) ? item.detections.filter(det => {
                    const className = det.class_name || 'unknown';
                    const color = det.dominant_color;

                    if (filterObjects.length > 0 && !filterObjects.includes(className)) return false;
                    if (filterColors.length > 0 && color && !filterColors.includes(color)) return false;

                    return true;
                  }) : [];

                  return (
                    <div
                      key={idx}
                      className="result-card glass-surface animate-fade-in"
                      style={{ animationDelay: `${(idx % 10) * 0.05}s` }}
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
                            e.target.onerror = null;
                            e.target.src = '/dummy.png';
                          }}
                          style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'cover' }}
                        />
                        <div className="result-card-overlay">
                          <div className="result-card-overlay-text">
                            <strong>{item.filename.length > 20 ? item.filename.substring(0, 20) + '...' : item.filename}</strong>
                            {relevantDetections.map((det, dIdx) => (
                              <span key={dIdx}>• {det.class_name || det.dominant_color || 'detection'} ({det.confidence ? (det.confidence * 100).toFixed(0) + '%' : 'N/A'})<br /></span>
                            ))}
                            {item.detections.length > relevantDetections.length && (
                              <span style={{ opacity: 0.7 }}>• + {item.detections.length - relevantDetections.length} other objects<br /></span>
                            )}
                          </div>
                        </div>
                      </div>
                      <div style={{ padding: '1rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                          <span style={{ fontSize: '0.9rem', fontWeight: 500 }} title={item.filename}>
                            {item.filename.length > 20 ? item.filename.substring(0, 20) + '...' : item.filename}
                          </span>
                        </div>
                        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                          {relevantDetections.map((det, dIdx) => (
                            <span key={dIdx} className="result-tag" title={`Confidence: ${det.confidence}`}>
                              {det.class_name || det.dominant_color || 'detection'}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>

            {/* Processing Log Panel */}
            {processingLog.length > 0 && (
              <div className="processing-log-panel glass-surface" style={{ margin: '0 1.5rem 1.5rem 1.5rem' }}>
                <div
                  className="processing-log-header"
                  onClick={() => setShowProcessingLog(!showProcessingLog)}
                >
                  <span>
                    <svg xmlns="http://www.w3.org/2000/svg" height="16px" viewBox="0 -960 960 960" width="16px" fill="currentColor" style={{ verticalAlign: 'middle', marginRight: '0.5rem' }}>
                      <path d="M160-160q-33 0-56.5-23.5T80-240v-480q0-33 23.5-56.5T160-800h640q33 0 56.5 23.5T880-720v480q0 33-23.5 56.5T800-160H160Zm0-80h640v-400H160v400Zm140-40-56-56 103-104-104-104 57-56 160 160-160 160Zm180 0v-80h240v80H480Z"/>
                    </svg>
                    Processing Log ({processingLog.length} entries)
                  </span>
                  <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="currentColor"
                    style={{ transform: showProcessingLog ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s ease' }}>
                    <path d="M480-345 240-585l56-56 184 184 184-184 56 56-240 240Z"/>
                  </svg>
                </div>
                {showProcessingLog && (
                  <div className="processing-log-body">
                    {processingLog.map((entry, idx) => (
                      <div key={idx} className="processing-log-entry">
                        <span className="log-time">{entry.time}</span>
                        <span className="log-message">{entry.message}</span>
                        {entry.progress > 0 && (
                          <span className="log-progress">{entry.progress}%</span>
                        )}
                      </div>
                    ))}
                    <div ref={logEndRef} />
                  </div>
                )}
              </div>
            )}
          </>
        )}
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
