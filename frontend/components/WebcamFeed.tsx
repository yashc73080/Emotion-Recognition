import { useEffect, useRef, useState } from 'react';
import { Camera, CameraOff } from 'lucide-react';

export default function WebcamFeed() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    async function setupCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsStreamActive(true);
          setError('');
        }
      } catch (err) {
        setError('Unable to access camera. Please make sure you have granted camera permissions.');
        setIsStreamActive(false);
      }
    }

    setupCamera();

    // Cleanup function to stop the stream when component unmounts
    return () => {
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 p-4">
      <div className="relative w-full max-w-4xl aspect-video bg-black rounded-lg overflow-hidden shadow-xl">
        {error ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white space-y-4">
            <CameraOff className="w-16 h-16" />
            <p className="text-lg text-center px-4">{error}</p>
          </div>
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full h-full object-cover"
            />
            {!isStreamActive && (
              <div className="absolute inset-0 flex items-center justify-center">
                <Camera className="w-16 h-16 text-white animate-pulse" />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}