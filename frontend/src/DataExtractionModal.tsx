import React, { useState, useEffect } from 'react';
import './DataExtractionModal.css';

interface DataExtractionModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface DataSource {
  name: string;
  url: string;
  source_type: string;
  extraction_method: string;
  update_frequency_hours: number;
  enabled: boolean;
  description?: string;
  priority?: string;
  extract_types?: string[];
}

interface ExtractionStats {
  total_items: number;
  source_stats: {
    [key: string]: {
      count: number;
      last_updated: string;
      status: string;
    };
  };
  type_stats: {
    [key: string]: number;
  };
  last_generated: string;
}

interface SearchResult {
  source_name: string;
  data_type: string;
  content: string;
  metadata: any;
  timestamp: string;
  url: string;
}

interface ExtractedData {
  source_name: string;
  data_type: string;
  content: string;
  timestamp: string;
  url: string;
}

interface ExtractionLog {
  source_name: string;
  status: string;
  message: string;
  timestamp: string;
}

const DataExtractionModal: React.FC<DataExtractionModalProps> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'sources' | 'search' | 'logs' | 'config'>('overview');
  const [stats, setStats] = useState<ExtractionStats | null>(null);
  const [sources, setSources] = useState<DataSource[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [recentData, setRecentData] = useState<ExtractedData[]>([]);
  const [logs, setLogs] = useState<ExtractionLog[]>([]);
  const [isExtracting, setIsExtracting] = useState(false);
  const [newSource, setNewSource] = useState<Partial<DataSource>>({
    name: '',
    url: '',
    source_type: 'github',
    extraction_method: 'repository_full',
    update_frequency_hours: 24,
    enabled: true,
    description: '',
    priority: 'medium'
  });

  const sourceTypes = [
    { value: 'github', label: 'GitHub Repository' },
    { value: 'huggingface_docs', label: 'HuggingFace Documentation' },
    { value: 'huggingface_org', label: 'HuggingFace Organization' },
    { value: 'huggingface_model', label: 'HuggingFace Model' },
    { value: 'web_scraping', label: 'Web Scraping' },
    { value: 'research_papers', label: 'Research Papers' }
  ];

  const extractionMethods = {
    github: ['repository_full', 'repository_code', 'repository_docs', 'repository_issues'],
    huggingface_docs: ['documentation_crawl', 'api_docs', 'tutorials'],
    huggingface_org: ['organization_models', 'organization_datasets'],
    huggingface_model: ['model_card', 'model_readme', 'model_config'],
    web_scraping: ['full_page', 'content_only', 'structured_data'],
    research_papers: ['abstract_only', 'full_text', 'metadata_only']
  };

  useEffect(() => {
    if (isOpen) {
      fetchStats();
      fetchSources();
      fetchRecentData();
      fetchLogs();
    }
  }, [isOpen]);

  const fetchStats = async () => {
    try {
      const response = await fetch('/api/data-extraction/stats');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Error fetching extraction stats:', error);
    }
  };

  const fetchSources = async () => {
    try {
      const response = await fetch('/api/data-extraction/sources');
      if (response.ok) {
        const data = await response.json();
        setSources(data.sources || []);
      }
    } catch (error) {
      console.error('Error fetching data sources:', error);
    }
  };

  const fetchRecentData = async () => {
    try {
      const response = await fetch('/api/data-extraction/recent?limit=10');
      if (response.ok) {
        const data = await response.json();
        setRecentData(data.recent_extractions || []);
      }
    } catch (error) {
      console.error('Error fetching recent data:', error);
    }
  };

  const fetchLogs = async () => {
    try {
      const response = await fetch('/api/data-extraction/logs?limit=20');
      if (response.ok) {
        const data = await response.json();
        setLogs(data.logs || []);
      }
    } catch (error) {
      console.error('Error fetching logs:', error);
    }
  };

  const runExtraction = async () => {
    setIsExtracting(true);
    try {
      const response = await fetch('/api/data-extraction/extract/run', {
        method: 'POST'
      });
      if (response.ok) {
        // Refresh data after extraction
        setTimeout(() => {
          fetchStats();
          fetchRecentData();
          fetchLogs();
        }, 2000);
      }
    } catch (error) {
      console.error('Error running extraction:', error);
    } finally {
      setIsExtracting(false);
    }
  };

  const searchData = async () => {
    if (!searchQuery.trim()) return;

    try {
      const response = await fetch('/api/data-extraction/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: searchQuery,
          limit: 20
        })
      });

      if (response.ok) {
        const data = await response.json();
        setSearchResults(data.results || []);
      }
    } catch (error) {
      console.error('Error searching data:', error);
    }
  };

  const exportData = async () => {
    try {
      const response = await fetch('/api/data-extraction/export', {
        method: 'POST'
      });
      if (response.ok) {
        const data = await response.json();
        alert(`Data exported successfully to: ${data.export_path}`);
      }
    } catch (error) {
      console.error('Error exporting data:', error);
    }
  };

  const addSource = async () => {
    if (!newSource.name || !newSource.url) return;

    try {
      const response = await fetch('/api/data-extraction/sources', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(newSource)
      });

      if (response.ok) {
        fetchSources();
        setNewSource({
          name: '',
          url: '',
          source_type: 'github',
          extraction_method: 'repository_full',
          update_frequency_hours: 24,
          enabled: true,
          description: '',
          priority: 'medium'
        });
      }
    } catch (error) {
      console.error('Error adding source:', error);
    }
  };

  const toggleSource = async (sourceName: string, enabled: boolean) => {
    try {
      const response = await fetch(`/api/data-extraction/sources/${sourceName}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ enabled })
      });

      if (response.ok) {
        fetchSources();
      }
    } catch (error) {
      console.error('Error toggling source:', error);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content data-extraction-modal">
        <div className="modal-header">
          <h2>Data Extraction System</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        <div className="extraction-tabs">
          <button 
            className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button 
            className={`tab ${activeTab === 'sources' ? 'active' : ''}`}
            onClick={() => setActiveTab('sources')}
          >
            Data Sources
          </button>
          <button 
            className={`tab ${activeTab === 'search' ? 'active' : ''}`}
            onClick={() => setActiveTab('search')}
          >
            Search & Browse
          </button>
          <button 
            className={`tab ${activeTab === 'logs' ? 'active' : ''}`}
            onClick={() => setActiveTab('logs')}
          >
            Extraction Logs
          </button>
          <button 
            className={`tab ${activeTab === 'config' ? 'active' : ''}`}
            onClick={() => setActiveTab('config')}
          >
            Configuration
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'overview' && (
            <div className="overview-tab">
              <div className="stats-section">
                <h3>Extraction Statistics</h3>
                {stats ? (
                  <div className="stats-grid">
                    <div className="stat-card">
                      <h4>Total Items</h4>
                      <div className="stat-value">{stats.total_items}</div>
                    </div>
                    <div className="stat-card">
                      <h4>Active Sources</h4>
                      <div className="stat-value">{Object.keys(stats.source_stats).length}</div>
                    </div>
                    <div className="stat-card">
                      <h4>Data Types</h4>
                      <div className="stat-value">{Object.keys(stats.type_stats).length}</div>
                    </div>
                    <div className="stat-card">
                      <h4>Last Updated</h4>
                      <div className="stat-value small">{new Date(stats.last_generated).toLocaleString()}</div>
                    </div>
                  </div>
                ) : (
                  <div className="loading">Loading statistics...</div>
                )}

                <div className="source-breakdown">
                  <h4>Data by Source</h4>
                  {stats?.source_stats && Object.entries(stats.source_stats).map(([source, data]) => (
                    <div key={source} className="source-stat">
                      <div className="source-name">{source}</div>
                      <div className="source-count">{data.count} items</div>
                      <div className={`source-status ${data.status.toLowerCase()}`}>
                        {data.status}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="type-breakdown">
                  <h4>Data by Type</h4>
                  {stats?.type_stats && Object.entries(stats.type_stats).map(([type, count]) => (
                    <div key={type} className="type-stat">
                      <span className="type-name">{type}</span>
                      <span className="type-count">{count}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="recent-data-section">
                <h3>Recent Extractions</h3>
                <div className="recent-data-list">
                  {recentData.map((item, index) => (
                    <div key={index} className="recent-item">
                      <div className="recent-header">
                        <span className="source-name">{item.source_name}</span>
                        <span className="data-type">{item.data_type}</span>
                        <span className="timestamp">{new Date(item.timestamp).toLocaleString()}</span>
                      </div>
                      <div className="content-preview">
                        {item.content.substring(0, 200)}...
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="actions-section">
                <button 
                  onClick={runExtraction}
                  disabled={isExtracting}
                  className="extraction-btn primary"
                >
                  {isExtracting ? 'Extracting...' : 'Run Extraction'}
                </button>
                <button onClick={exportData} className="export-btn">
                  Export Data
                </button>
              </div>
            </div>
          )}

          {activeTab === 'sources' && (
            <div className="sources-tab">
              <div className="sources-header">
                <h3>Data Sources</h3>
                <button 
                  onClick={fetchSources}
                  className="refresh-btn"
                >
                  Refresh
                </button>
              </div>

              <div className="sources-list">
                {sources.map((source, index) => (
                  <div key={index} className={`source-card ${source.enabled ? 'enabled' : 'disabled'}`}>
                    <div className="source-header">
                      <h4>{source.name}</h4>
                      <div className="source-controls">
                        <label className="toggle-switch">
                          <input 
                            type="checkbox"
                            checked={source.enabled}
                            onChange={(e) => toggleSource(source.name, e.target.checked)}
                          />
                          <span className="toggle-slider"></span>
                        </label>
                      </div>
                    </div>
                    
                    <div className="source-details">
                      <div className="source-url">{source.url}</div>
                      <div className="source-type">{source.source_type}</div>
                      <div className="source-method">{source.extraction_method}</div>
                      <div className="source-frequency">Updates every {source.update_frequency_hours}h</div>
                      {source.description && (
                        <div className="source-description">{source.description}</div>
                      )}
                    </div>

                    {source.extract_types && (
                      <div className="extract-types">
                        {source.extract_types.map((type, i) => (
                          <span key={i} className="extract-type-tag">{type}</span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <div className="add-source-section">
                <h4>Add New Source</h4>
                <div className="add-source-form">
                  <input 
                    type="text"
                    placeholder="Source name"
                    value={newSource.name || ''}
                    onChange={(e) => setNewSource(prev => ({ ...prev, name: e.target.value }))}
                  />
                  <input 
                    type="url"
                    placeholder="Source URL"
                    value={newSource.url || ''}
                    onChange={(e) => setNewSource(prev => ({ ...prev, url: e.target.value }))}
                  />
                  <select 
                    value={newSource.source_type || 'github'}
                    onChange={(e) => setNewSource(prev => ({ 
                      ...prev, 
                      source_type: e.target.value,
                      extraction_method: extractionMethods[e.target.value as keyof typeof extractionMethods][0]
                    }))}
                  >
                    {sourceTypes.map(type => (
                      <option key={type.value} value={type.value}>{type.label}</option>
                    ))}
                  </select>
                  <select 
                    value={newSource.extraction_method || ''}
                    onChange={(e) => setNewSource(prev => ({ ...prev, extraction_method: e.target.value }))}
                  >
                    {newSource.source_type && extractionMethods[newSource.source_type as keyof typeof extractionMethods]?.map(method => (
                      <option key={method} value={method}>{method.replace('_', ' ')}</option>
                    ))}
                  </select>
                  <input 
                    type="number"
                    placeholder="Update frequency (hours)"
                    value={newSource.update_frequency_hours || 24}
                    onChange={(e) => setNewSource(prev => ({ ...prev, update_frequency_hours: parseInt(e.target.value) }))}
                    min="1"
                  />
                  <textarea 
                    placeholder="Description (optional)"
                    value={newSource.description || ''}
                    onChange={(e) => setNewSource(prev => ({ ...prev, description: e.target.value }))}
                    rows={2}
                  />
                  <button onClick={addSource} disabled={!newSource.name || !newSource.url}>
                    Add Source
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'search' && (
            <div className="search-tab">
              <div className="search-section">
                <h3>Search Extracted Data</h3>
                <div className="search-form">
                  <input 
                    type="text"
                    placeholder="Search for content, concepts, or keywords..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && searchData()}
                  />
                  <button onClick={searchData} disabled={!searchQuery.trim()}>
                    Search
                  </button>
                </div>

                {searchResults.length > 0 && (
                  <div className="search-results">
                    <h4>Search Results ({searchResults.length})</h4>
                    {searchResults.map((result, index) => (
                      <div key={index} className="search-result">
                        <div className="result-header">
                          <span className="result-source">{result.source_name}</span>
                          <span className="result-type">{result.data_type}</span>
                          <span className="result-timestamp">{new Date(result.timestamp).toLocaleString()}</span>
                        </div>
                        <div className="result-content">
                          {result.content.substring(0, 300)}...
                        </div>
                        <div className="result-url">
                          <a href={result.url} target="_blank" rel="noopener noreferrer">
                            View Source
                          </a>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="browse-section">
                <h3>Browse by Category</h3>
                <div className="category-buttons">
                  <button onClick={() => setSearchQuery('source_code')}>Source Code</button>
                  <button onClick={() => setSearchQuery('documentation')}>Documentation</button>
                  <button onClick={() => setSearchQuery('model_card')}>Model Cards</button>
                  <button onClick={() => setSearchQuery('research')}>Research</button>
                  <button onClick={() => setSearchQuery('examples')}>Examples</button>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'logs' && (
            <div className="logs-tab">
              <div className="logs-header">
                <h3>Extraction Logs</h3>
                <button onClick={fetchLogs} className="refresh-btn">
                  Refresh
                </button>
              </div>

              <div className="logs-list">
                {logs.map((log, index) => (
                  <div key={index} className={`log-entry ${log.status.toLowerCase()}`}>
                    <div className="log-header">
                      <span className="log-source">{log.source_name}</span>
                      <span className={`log-status ${log.status.toLowerCase()}`}>
                        {log.status}
                      </span>
                      <span className="log-timestamp">
                        {new Date(log.timestamp).toLocaleString()}
                      </span>
                    </div>
                    {log.message && (
                      <div className="log-message">{log.message}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'config' && (
            <div className="config-tab">
              <div className="config-section">
                <h3>System Configuration</h3>
                
                <div className="config-option">
                  <h4>Extraction Frequency</h4>
                  <p>How often the system should check for new data</p>
                  <select defaultValue="1">
                    <option value="1">Every hour</option>
                    <option value="6">Every 6 hours</option>
                    <option value="12">Every 12 hours</option>
                    <option value="24">Daily</option>
                  </select>
                </div>

                <div className="config-option">
                  <h4>Storage Settings</h4>
                  <p>Configure data storage and cleanup policies</p>
                  <label>
                    <input type="checkbox" defaultChecked />
                    Auto-cleanup old data (older than 30 days)
                  </label>
                  <label>
                    <input type="checkbox" defaultChecked />
                    Compress stored documents
                  </label>
                </div>

                <div className="config-option">
                  <h4>Extraction Depth</h4>
                  <p>Control how much content to extract from each source</p>
                  <select defaultValue="balanced">
                    <option value="minimal">Minimal (metadata only)</option>
                    <option value="balanced">Balanced (key content)</option>
                    <option value="comprehensive">Comprehensive (full content)</option>
                  </select>
                </div>

                <div className="config-option">
                  <h4>API Limits</h4>
                  <p>Configure rate limiting for external APIs</p>
                  <label>
                    GitHub API requests per hour: 
                    <input type="number" defaultValue={100} min={10} max={1000} />
                  </label>
                  <label>
                    HuggingFace API requests per hour: 
                    <input type="number" defaultValue={200} min={10} max={1000} />
                  </label>
                </div>

                <div className="config-actions">
                  <button className="save-config-btn">Save Configuration</button>
                  <button className="reset-config-btn">Reset to Defaults</button>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
};

export default DataExtractionModal;
