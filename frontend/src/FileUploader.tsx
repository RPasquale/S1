import { ChangeEvent, useRef } from 'react';

interface Props {
  onUpload: (files: FileList) => void;
}

const FileUploader = ({ onUpload }: Props) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) onUpload(e.target.files);
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
        webkitdirectory="true"
        multiple
        onChange={handleChange}
      />
      <button className="upload-button" onClick={handleClick}>
        Choose folder
      </button>
    </div>
  );
};

export default FileUploader;