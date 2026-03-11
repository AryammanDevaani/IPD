import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, X, FileText } from "lucide-react";

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  isVerifying: boolean;
  onClear: () => void;
}

const UploadZone = ({ onFileSelect, isVerifying, onClear }: UploadZoneProps) => {
  const [file, setFile] = useState<File | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const selectedFile = acceptedFiles[0];
      setFile(selectedFile);
      onFileSelect(selectedFile);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    disabled: isVerifying,
    multiple: false,
    accept: {
      "application/pdf": [".pdf"],
      "image/*": [".png", ".jpg", ".jpeg"],
      "application/msword": [".doc"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    },
  });

  const clearFile = () => {
    setFile(null);
    onClear();
  };

  return (
    <div className="relative group">
      {/* Underlying glow for drag state */}
      <div className={`absolute -inset-1 rounded-[2rem] blur-xl transition-all duration-500 opacity-0 bg-cyan-500/30 group-hover:opacity-100 ${isDragActive ? 'opacity-100 bg-cyan-400/50 animate-pulse' : ''}`} />
      
      <div {...getRootProps()} className="relative z-10 w-full">
        <input {...getInputProps()} />
        <motion.div
          className={`relative cursor-pointer overflow-hidden rounded-2xl border-2 border-dashed transition-all duration-300 ${
            isDragActive ? "border-cyan-400 bg-cyan-500/10 scale-[1.02]" : "border-slate-700 bg-slate-900 hover:border-cyan-500/50 hover:bg-slate-800/80"
          } ${isVerifying ? "cursor-not-allowed opacity-60 grayscale-[50%]" : ""}`}
        >
          <AnimatePresence mode="wait">
          {file ? (
            <motion.div key="file" className="flex flex-col items-center gap-4 p-8">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-cyan-500/20">
                <FileText className="h-8 w-8 text-cyan-500" />
              </div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm text-slate-300">{file.name}</span>
                {!isVerifying && (
                  <button onClick={(e) => { e.stopPropagation(); clearFile(); }} className="rounded-full p-1 hover:bg-red-500/20 hover:text-red-500 text-slate-400">
                    <X className="h-4 w-4" />
                  </button>
                )}
              </div>
            </motion.div>
          ) : (
            <motion.div key="empty" className="flex flex-col items-center gap-4 p-8 text-slate-300 group-hover:-translate-y-1 transition-transform duration-500">
              <div className="rounded-2xl bg-slate-800 border border-white/5 p-5 shadow-lg relative">
                <div className="absolute inset-0 bg-cyan-500/10 rounded-2xl blur-md opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                <Upload className="h-10 w-10 text-cyan-400 relative z-10 group-hover:scale-110 transition-transform duration-500" />
              </div>
              <div className="text-center">
                <p className="font-semibold text-slate-100 text-lg">Drop your document here</p>
                <p className="mt-1 text-sm text-slate-400">or click to browse • Images, PDFs, Documents</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        </motion.div>
      </div>
    </div>
  );
};

export default UploadZone;
