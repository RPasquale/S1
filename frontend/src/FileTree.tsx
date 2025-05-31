import React, { useState } from 'react';
import './FileTree.css';

interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'folder';
  children?: FileNode[];
  size?: number;
  lastModified?: number;
}

interface Props {
  files: FileNode[];
  onFileSelect?: (file: FileNode) => void;
}

const FileTreeNode: React.FC<{
  node: FileNode;
  level: number;
  onFileSelect?: (file: FileNode) => void;
}> = ({ node, level, onFileSelect }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const hasChildren = node.children && node.children.length > 0;

  const handleToggle = () => {
    if (hasChildren) {
      setIsExpanded(!isExpanded);
    }
    if (onFileSelect) {
      onFileSelect(node);
    }
  };

  const getFileIcon = (fileName: string, isFolder: boolean) => {
    if (isFolder) {
      return isExpanded ? '📂' : '📁';
    }
    
    const ext = fileName.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'pdf': return '📄';
      case 'txt': return '📝';
      case 'doc':
      case 'docx': return '📃';
      case 'xls':
      case 'xlsx': return '📊';
      case 'ppt':
      case 'pptx': return '📽️';
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif': return '🖼️';
      case 'zip':
      case 'rar': return '🗜️';
      default: return '📄';
    }
  };

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return '';
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="file-tree-node">
      <div 
        className={`file-tree-item ${node.type}`}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
        onClick={handleToggle}
      >
        <span className="file-icon">
          {getFileIcon(node.name, node.type === 'folder')}
        </span>
        <span className="file-name">{node.name}</span>
        {node.type === 'file' && node.size && (
          <span className="file-size">{formatFileSize(node.size)}</span>
        )}
        {hasChildren && (
          <span className={`expand-icon ${isExpanded ? 'expanded' : ''}`}>
            ▶
          </span>
        )}
      </div>
      
      {isExpanded && hasChildren && (
        <div className="file-tree-children">
          {node.children!.map((child, index) => (
            <FileTreeNode
              key={`${child.path}-${index}`}
              node={child}
              level={level + 1}
              onFileSelect={onFileSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
};

const FileTree: React.FC<Props> = ({ files, onFileSelect }) => {
  const [searchTerm, setSearchTerm] = useState('');

  const filterFiles = (nodes: FileNode[], term: string): FileNode[] => {
    if (!term) return nodes;
    
    return nodes.reduce((filtered: FileNode[], node) => {
      if (node.name.toLowerCase().includes(term.toLowerCase())) {
        filtered.push(node);
      } else if (node.children) {
        const filteredChildren = filterFiles(node.children, term);
        if (filteredChildren.length > 0) {
          filtered.push({
            ...node,
            children: filteredChildren
          });
        }
      }
      return filtered;
    }, []);
  };

  const filteredFiles = filterFiles(files, searchTerm);

  return (
    <div className="file-tree">
      <div className="file-tree-header">
        <h3>📁 Uploaded Files</h3>
        <div className="file-search">
          <input
            type="text"
            placeholder="Search files..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="file-search-input"
          />
        </div>
      </div>
      
      <div className="file-tree-content">
        {filteredFiles.length === 0 ? (
          <div className="no-files">
            {searchTerm ? 'No files match your search' : 'No files uploaded yet'}
          </div>
        ) : (
          filteredFiles.map((file, index) => (
            <FileTreeNode
              key={`${file.path}-${index}`}
              node={file}
              level={0}
              onFileSelect={onFileSelect}
            />
          ))
        )}
      </div>
    </div>
  );
};

export default FileTree;
