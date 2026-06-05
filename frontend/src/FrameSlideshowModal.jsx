import React, { useState, useEffect, useRef } from 'react';

const FrameSlideshowModal = ({
  isOpen,
  onClose,
  frameItem,
  tracker,
  API_BASE,
  fps,
  interval,
  videoPath,
  outputDir
}) => {
  if (!isOpen || !frameItem) return null;

  const [contextFrames, setContextFrames] = useState(15);
  const [forwardFrames, setForwardFrames] = useState(30);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(2); // fps

  const [loadedFrames, setLoadedFrames] = useState([]);

  const timerRef = useRef(null);

  const baseDir = tracker === 'bytetrack' ? 'extracted_frames_b' : 'extracted_frames_yx';

  // Helper to parse frame number
  const getFrameNumber = (filename) => {
    if (!filename) return 0;
    const match = filename.match(/frame_(\d+)\.jpg/);
    return match ? parseInt(match[1], 10) : 0;
  };

  const centerFrameNumber = getFrameNumber(frameItem.filename);

  useEffect(() => {
    // Generate the array of frame numbers
    const start = Math.max(0, centerFrameNumber - contextFrames);
    const end = centerFrameNumber + forwardFrames;

    const frames = [];

    // Normalize outputDir path
    const normalizedOutputDir = outputDir ? outputDir.replace(/\/+$/, '') : './outputs';

    for (let i = start; i <= end; i++) {
      const paddedNum = i.toString().padStart(6, '0');
      const frameFilename = `frame_${paddedNum}.jpg`;
      const fullPath = `${normalizedOutputDir}/${baseDir}/${frameFilename}`;
      const foundPath = `${normalizedOutputDir}/found_frames/${frameFilename}`;

      frames.push({
        number: i,
        filename: frameFilename,
        url: `${API_BASE}/api/file?path=${encodeURIComponent(fullPath)}`,
        foundUrl: `${API_BASE}/api/file?path=${encodeURIComponent(foundPath)}`,
        isAnnotated: i === centerFrameNumber,
      });
    }

    setLoadedFrames(frames);

    // Set slider to the center frame
    const centerIndex = frames.findIndex(f => f.number === centerFrameNumber);
    setCurrentFrameIndex(centerIndex >= 0 ? centerIndex : 0);
    setIsPlaying(false);
  }, [centerFrameNumber, contextFrames, forwardFrames, API_BASE, baseDir]);

  // Handle Playback
  useEffect(() => {
    if (isPlaying && loadedFrames.length > 0) {
      const msPerFrame = 1000 / playbackSpeed;
      timerRef.current = setInterval(() => {
        setCurrentFrameIndex((prev) => {
          if (prev >= loadedFrames.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, msPerFrame);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isPlaying, playbackSpeed, loadedFrames.length]);

  // Keyboard shortcuts for modal
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key === 'ArrowLeft') {
        setCurrentFrameIndex(prev => Math.max(0, prev - 1));
        setIsPlaying(false);
      } else if (e.key === 'ArrowRight') {
        setCurrentFrameIndex(prev => Math.min(loadedFrames.length - 1, prev + 1));
        setIsPlaying(false);
      } else if (e.key === ' ') {
        e.preventDefault();
        setIsPlaying(prev => !prev);
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [loadedFrames.length]);

  const currentFrame = loadedFrames[currentFrameIndex] || null;

  // Formatting timestamp
  const calculateTimestamp = (frameNum) => {
    const validFps = (fps && fps !== 'unknown' && fps > 0) ? fps : 30;
    const seconds = frameNum * (interval / validFps);
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
  };

  const handleExport = () => {
    if (!currentFrame) return;
    const link = document.createElement('a');
    link.href = currentFrame.url;
    link.download = currentFrame.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleOpenSystemPlayer = async () => {
    if (!currentFrame) return;
    const validFps = (fps && fps !== 'unknown' && fps > 0) ? fps : 30;
    const timestampSec = currentFrame.number * (interval / validFps);

    const formData = new FormData();
    formData.append('video_path', videoPath);
    formData.append('timestamp', timestampSec);

    try {
      await fetch(`${API_BASE}/api/open-player`, {
        method: 'POST',
        body: formData
      });
    } catch (e) {
      console.error("Failed to open player", e);
      alert("Failed to connect to the backend to open the system player.");
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()} style={{ width: '80%', maxWidth: '900px', backgroundColor: 'var(--bg-surface)', padding: '1.5rem', borderRadius: '8px', boxShadow: '0 10px 30px rgba(0,0,0,0.5)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h2 style={{ margin: 0, fontSize: '1.2rem' }}>Frame Slideshow</h2>
          <button onClick={onClose} style={{ background: 'none', border: 'none', fontSize: '1.5rem', cursor: 'pointer', color: 'var(--text-secondary)' }}>&times;</button>
        </div>

        {/* Image Area */}
        <div style={{ position: 'relative', width: '100%', backgroundColor: 'black', borderRadius: '4px', overflow: 'hidden', aspectRatio: '16/9', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          {currentFrame && (
            <img
              src={currentFrame.isAnnotated ? currentFrame.foundUrl : currentFrame.url}
              alt="Frame"
              style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
              onError={(e) => { e.target.src = '/dummy.png'; }}
            />
          )}
        </div>

        {/* Info Bar */}
        {currentFrame && (
          <div style={{ textAlign: 'center', margin: '1rem 0', color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
            Frame: {currentFrameIndex + 1}/{loadedFrames.length} - #{currentFrame.number} {currentFrame.isAnnotated ? '(Annotated)' : ''} - Time: {calculateTimestamp(currentFrame.number)}
          </div>
        )}

        {/* Scrubber */}
        <div style={{ margin: '1rem 0' }}>
          <input
            type="range"
            min="0"
            max={loadedFrames.length > 0 ? loadedFrames.length - 1 : 0}
            value={currentFrameIndex}
            onChange={(e) => {
              setCurrentFrameIndex(parseInt(e.target.value, 10));
              setIsPlaying(false);
            }}
            style={{ width: '100%', accentColor: 'var(--primary-accent)' }}
          />
        </div>

        {/* Controls Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>

          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <button className="btn btn-primary" onClick={() => setIsPlaying(!isPlaying)} style={{ flex: 1 }}>
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            <select
              className="input-field"
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
              style={{ flex: 1 }}
            >
              <option value="1">Slow (1 fps)</option>
              <option value="2">Medium (2 fps)</option>
              <option value="5">Fast (5 fps)</option>
            </select>
          </div>

          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <div className="input-group" style={{ marginBottom: 0, flex: 1, flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
              <label className="input-label" style={{ marginBottom: 0 }}>Context Frames:</label>
              <input type="number" className="input-field" value={contextFrames} onChange={e => setContextFrames(Number(e.target.value))} min="0" max="100" />
            </div>
            <div className="input-group" style={{ marginBottom: 0, flex: 1, flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
              <label className="input-label" style={{ marginBottom: 0 }}>Forward Frames:</label>
              <input type="number" className="input-field" value={forwardFrames} onChange={e => setForwardFrames(Number(e.target.value))} min="0" max="100" />
            </div>
          </div>

          <button className="btn btn-secondary" onClick={handleExport}>
            Export Frame
          </button>

          <button className="btn btn-secondary" onClick={handleOpenSystemPlayer}>
            Open in System Player
          </button>

        </div>
      </div>

      {/* Basic modal styling since we are adding inline, we can put some global styles here or in index.css */}
      <style dangerouslySetInnerHTML={{
        __html: `
        .modal-overlay {
          position: fixed;
          top: 0; left: 0; right: 0; bottom: 0;
          background: rgba(0,0,0,0.7);
          display: flex;
          justify-content: center;
          align-items: center;
          z-index: 1000;
        }
      `}} />
    </div>
  );
};

export default FrameSlideshowModal;
