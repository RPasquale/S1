import { ChangeEvent, useRef } from 'react';

interface Props {
  onUpload: (files: FileList) => void;
}

const FileUploader = ({ onUpload }: Props) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    console.log('File input changed:', e.target.files);
    if (e.target.files && e.target.files.length > 0) {
      console.log('Selected files:', Array.from(e.target.files).map(f => f.name));
      onUpload(e.target.files);
    } else {
      console.log('No files selected');
    }
  };

  const handleFolderChange = (e: ChangeEvent<HTMLInputElement>) => {
    console.log('Folder input changed:', e.target.files);
    if (e.target.files && e.target.files.length > 0) {
      console.log('Selected folder files:', Array.from(e.target.files).map(f => f.name));
      onUpload(e.target.files);
    } else {
      console.log('No folder files selected');
    }
  };

  const handleFileClick = () => {
    console.log('Upload files button clicked');
    fileInputRef.current?.click();
  };

  const handleFolderClick = () => {
    console.log('Upload folder button clicked');
    folderInputRef.current?.click();
  };

  return (
    <div className="file-uploader">
      <label>Upload PDF documents:</label>
      
      {/* Individual files */}
      <input
        ref={fileInputRef}
        id="file-input"
        type="file"
        accept=".pdf"
        multiple
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />
      <button className="upload-button" onClick={handleFileClick}>
        Choose PDF Files
      </button>
      
      {/* Folder */}
      <input
        ref={folderInputRef}
        id="folder-input"
        type="file"
        webkitdirectory="true"
        multiple
        onChange={handleFolderChange}
        style={{ display: 'none' }}
      />
      <button className="upload-button" onClick={handleFolderClick}>
        Choose Folder
      </button>
    </div>
  );
};

export default FileUploader;