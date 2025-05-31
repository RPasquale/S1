import { ChangeEvent, useRef } from 'react';

interface Props {
  onUpload: (files: FileList) => void;
}

// Extend the HTML input element to include webkitdirectory
declare module 'react' {
  interface InputHTMLAttributes<T> extends HTMLAttributes<T> {
    webkitdirectory?: string;
  }
}

const FileUploader = ({ onUpload }: Props) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onUpload(e.target.files);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="file-uploader">
      <label>Upload PDF folder:</label>
      <input
        ref={fileInputRef}
        id="folder-input"
        type="file"
        webkitdirectory=""
        multiple
        onChange={handleChange}
        style={{ display: 'none' }}
      />
      <button className="upload-button" onClick={handleClick}>
        Choose folder
      </button>
    </div>
  );
};

export default FileUploader;