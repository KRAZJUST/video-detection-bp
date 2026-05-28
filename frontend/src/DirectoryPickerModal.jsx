import React, { useState, useEffect } from 'react';

const DirectoryPickerModal = ({ isOpen, onClose, onSelect, initialPath, API_BASE }) => {
  const [currentPath, setCurrentPath] = useState(initialPath || '.');
  const [directories, setDirectories] = useState([]);
  const [parentPath, setParentPath] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchDirectory = async (path) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/list-directory?path=${encodeURIComponent(path)}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to load directory');
      setCurrentPath(data.current_path);
      setParentPath(data.parent_path);
      setDirectories(data.directories);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchDirectory(initialPath || '.');
    }
  }, [isOpen, initialPath]);

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
      display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000
    }}>
      <div className="glass-surface" style={{ width: '90%', maxWidth: '600px', display: 'flex', flexDirection: 'column', maxHeight: '80vh' }}>
        <div style={{ padding: '1.5rem', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ margin: 0, fontSize: '1.2rem', color: 'var(--text-primary)' }}>Select Output Directory</h3>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '1.5rem' }}>&times;</button>
        </div>
        
        <div style={{ padding: '1rem 1.5rem', background: 'var(--bg-surface-elevated)', borderBottom: '1px solid var(--border-color)', fontSize: '0.9rem', wordBreak: 'break-all', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <strong>Current:</strong> <span style={{ color: 'var(--primary-accent)', fontFamily: 'monospace' }}>{currentPath}</span>
        </div>
        
        <div style={{ padding: '0.5rem', overflowY: 'auto', flex: 1, minHeight: '300px' }}>
          {error && <div style={{ color: 'var(--danger)', padding: '1rem', textAlign: 'center' }}>{error}</div>}
          
          {loading ? (
            <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>Loading...</div>
          ) : (
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              {parentPath && (
                <li 
                  onClick={() => fetchDirectory(parentPath)}
                  style={{ padding: '0.75rem 1rem', cursor: 'pointer', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '2px', background: 'transparent', transition: 'background 0.2s' }}
                  onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-surface-elevated)'}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                >
                  <span style={{ fontSize: '1.2rem' }}>📁</span> 
                  <span style={{ fontWeight: '500' }}>.. (Parent Directory)</span>
                </li>
              )}
              {directories.map((dir, idx) => (
                <li 
                  key={idx}
                  onClick={() => fetchDirectory(`${currentPath}/${dir}`)}
                  style={{ padding: '0.75rem 1rem', cursor: 'pointer', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '2px', background: 'transparent', transition: 'background 0.2s' }}
                  onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-surface-elevated)'}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                >
                  <span style={{ fontSize: '1.2rem' }}>📁</span> 
                  <span>{dir}</span>
                </li>
              ))}
              {directories.length === 0 && !error && (
                <li style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>No subdirectories found.</li>
              )}
            </ul>
          )}
        </div>
        
        <div style={{ padding: '1.25rem 1.5rem', borderTop: '1px solid var(--border-color)', display: 'flex', justifyContent: 'flex-end', gap: '1rem', background: 'var(--bg-surface-elevated)', borderBottomLeftRadius: '12px', borderBottomRightRadius: '12px' }}>
          <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={() => { onSelect(currentPath); onClose(); }}>Select This Folder</button>
        </div>
      </div>
    </div>
  );
};

export default DirectoryPickerModal;
