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
                  <span style={{ fontSize: '1.2rem', display: 'flex', alignItems: 'center' }}>
                    <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="currentColor"><path d="M160-160q-33 0-56.5-23.5T80-240v-480q0-33 23.5-56.5T160-800h240l80 80h320q33 0 56.5 23.5T880-640v242q-18-14-38-23t-42-19v-200H447l-80-80H160v480h120v80H160ZM640-40q-91 0-168-48T360-220q35-84 112-132t168-48q91 0 168 48t112 132q-35 84-112 132T640-40Zm107.5-106q50.5-26 82.5-74-32-48-82.5-74T640-320q-57 0-107.5 26T450-220q32 48 82.5 74T640-120q57 0 107.5-26Zm-150-31.5Q580-195 580-220t17.5-42.5Q615-280 640-280t42.5 17.5Q700-245 700-220t-17.5 42.5Q665-160 640-160t-42.5-17.5ZM160-240v-480 277-37 240Z" /></svg>
                  </span>
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
                  <span style={{ fontSize: '1.2rem', display: 'flex', alignItems: 'center' }}>
                    <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="currentColor"><path d="M160-160q-33 0-56.5-23.5T80-240v-480q0-33 23.5-56.5T160-800h240l80 80h320q33 0 56.5 23.5T880-640v242q-18-14-38-23t-42-19v-200H447l-80-80H160v480h120v80H160ZM640-40q-91 0-168-48T360-220q35-84 112-132t168-48q91 0 168 48t112 132q-35 84-112 132T640-40Zm107.5-106q50.5-26 82.5-74-32-48-82.5-74T640-320q-57 0-107.5 26T450-220q32 48 82.5 74T640-120q57 0 107.5-26Zm-150-31.5Q580-195 580-220t17.5-42.5Q615-280 640-280t42.5 17.5Q700-245 700-220t-17.5 42.5Q665-160 640-160t-42.5-17.5ZM160-240v-480 277-37 240Z" /></svg>
                  </span>
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
