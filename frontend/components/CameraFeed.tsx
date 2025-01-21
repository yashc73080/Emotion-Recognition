'use client'; 

import { useEffect, useRef, useState } from 'react';
import { Camera, XCircle } from 'lucide-react';

const CameraFeed = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState<string>('');
  const [isStreaming, setIsStreaming] = useState(false);

  useEffect(() => {
    async function setupCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsStreaming(true);
        }
      } catch (err) {
        setError('Failed to access camera. Please make sure you have granted camera permissions.');
        console.error('Error accessing camera:', err);
      }
    }

    setupCamera();

    // Cleanup function
    return () => {
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 p-4">
      <div className="relative w-full max-w-4xl rounded-lg overflow-hidden shadow-xl">
        {error ? (
          <div className="bg-red-500 p-6 rounded-lg text-white flex items-center gap-2">
            <XCircle className="h-6 w-6" />
            <span>{error}</span>
          </div>
        ) : !isStreaming ? (
          <div className="bg-gray-800 p-6 rounded-lg text-white flex items-center gap-2">
            <Camera className="h-6 w-6 animate-pulse" />
            <span>Connecting to camera...</span>
          </div>
        ) : (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full h-full rounded-lg"
          />
        )}
      </div>
    </div>
  );
};

export default CameraFeed;