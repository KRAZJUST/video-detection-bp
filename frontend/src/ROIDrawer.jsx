import React, { useState, useRef, useEffect } from 'react';

const ROIDrawer = ({ width, height, onROIChange }) => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [roi, setRoi] = useState(null); // {x, y, w, h}
  const [isShiftDown, setIsShiftDown] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Shift') setIsShiftDown(true);
    };
    const handleKeyUp = (e) => {
      if (e.key === 'Shift') setIsShiftDown(false);
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  // Draw the ROI rectangle on the canvas
  const drawROI = (ctx, rect) => {
    ctx.clearRect(0, 0, width, height);
    if (!rect) return;

    // Draw semi-transparent dark overlay over the whole canvas
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(0, 0, width, height);

    // Clear out the center (the ROI)
    ctx.clearRect(rect.x, rect.y, rect.w, rect.h);

    // Draw red border around the ROI
    ctx.strokeStyle = '#ff4d4f';
    ctx.lineWidth = 2;
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      drawROI(ctx, roi);
    }
  }, [roi, width, height]);

  const getMousePos = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  };

  const handleMouseDown = (e) => {
    const pos = getMousePos(e);
    setStartPos(pos);
    setIsDrawing(true);
    setRoi(null);
    onROIChange(null);
  };

  const handleMouseMove = (e) => {
    if (!isDrawing) return;
    const currentPos = getMousePos(e);

    const x = Math.min(startPos.x, currentPos.x);
    const y = Math.min(startPos.y, currentPos.y);
    const w = Math.abs(currentPos.x - startPos.x);
    const h = Math.abs(currentPos.y - startPos.y);

    const newRoi = { x, y, w, h };

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    drawROI(ctx, newRoi);
  };

  const handleMouseUp = (e) => {
    if (!isDrawing) return;
    setIsDrawing(false);

    const currentPos = getMousePos(e);
    const x = Math.min(startPos.x, currentPos.x);
    const y = Math.min(startPos.y, currentPos.y);
    const w = Math.abs(currentPos.x - startPos.x);
    const h = Math.abs(currentPos.y - startPos.y);

    // Ignore tiny accidental clicks
    if (w < 10 || h < 10) {
      setRoi(null);
      onROIChange(null);
      return;
    }

    const finalRoi = { x, y, w, h };
    setRoi(finalRoi);
    // Convert to [x1, y1, x2, y2]
    onROIChange([Math.round(x), Math.round(y), Math.round(x + w), Math.round(y + h)]);
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', display: 'block', pointerEvents: 'none' }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          cursor: isShiftDown ? 'crosshair' : 'default',
          zIndex: 10,
          pointerEvents: isShiftDown ? 'auto' : 'none',
          borderRadius: '8px'
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />

      <div style={{ position: 'absolute', top: 10, right: 10, zIndex: 11, display: 'flex', gap: '0.5rem', pointerEvents: 'auto' }}>
        {roi && (
          <button
            onClick={() => { setRoi(null); onROIChange(null); }}
            style={{
              background: 'var(--danger)',
              color: 'white',
              border: 'none',
              padding: '5px 10px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '12px',
              boxShadow: '0 2px 5px rgba(0,0,0,0.2)'
            }}
          >
            Clear ROI
          </button>
        )}
      </div>
    </div>
  );
};

export default ROIDrawer;
