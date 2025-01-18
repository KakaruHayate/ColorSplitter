import React, { useState, useCallback } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid } from 'recharts';
import { Play, Pause } from 'lucide-react';

interface DataPoint {
    filename: string;
    clust: number;
    x: number;
    y: number;
}


const AudioVisualizer = () => {
    const [data, setData] = useState<DataPoint[]>([]);
    const [selectedFile, setSelectedFile] = useState<string | null>(null);


    const [currentAudio, setCurrentAudio] = useState<{
        file: string;
        audio: HTMLAudioElement | null;
        isPlaying: boolean;
        progress: number;
    } | null>(null);

    const handlePlay = (filename: string) => {
        if (currentAudio?.file === filename) {
            // Resume/pause existing audio
            if (currentAudio.isPlaying) {
                currentAudio.audio?.pause();
                setCurrentAudio(prev => prev ? { ...prev, isPlaying: false } : null);
            } else {
                currentAudio.audio?.play();
                setCurrentAudio(prev => prev ? { ...prev, isPlaying: true } : null);
            }
        } else {
            // Stop previous audio if any
            currentAudio?.audio?.pause();

            // Create new audio
            const audio = new Audio(`http://localhost:8080/${filename}`);
            audio.addEventListener('ended', () => {
                setCurrentAudio(prev => prev ? { ...prev, isPlaying: false, progress: 0 } : null);
            });
            audio.play();
            setCurrentAudio({
                file: filename,
                audio,
                isPlaying: true,
                progress: 0
            });
        }
    };


    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file && file.name.endsWith('.csv')) {
            const reader = new FileReader();
            reader.onload = (event) => {
                const text = event.target?.result as string;
                const lines = text.split('\n');
                const parsedData: DataPoint[] = lines.slice(1)
                    .filter(line => line.trim())
                    .map(line => {
                        const values = line.split(',');
                        return {
                            filename: values[0],
                            clust: parseInt(values[1], 10),
                            x: parseFloat(values[2]),
                            y: parseFloat(values[3])
                        };
                    });
                setData(parsedData);
            };
            reader.readAsText(file);
        }
    }, []);

    const handleClick = (point: DataPoint) => {
        setSelectedFile(point.filename);
    };

    const handleReset = () => {
        setSelectedFile(null);
    };

    const displayFiles = selectedFile ? data.filter(d => d.filename === selectedFile) : data;

    return (
        <div className="flex h-screen p-4 gap-4">
            <div className="flex-1">
                <div
                    className="w-full h-32 border-2 border-dashed border-gray-300 rounded-lg mb-4 flex items-center justify-center bg-gray-50"
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                >
                    <p className="text-gray-500">Drop CSV here</p>
                </div>

                {data.length > 0 && (
                    <ScatterChart
                        width={800}
                        height={600}
                        margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                    >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" dataKey="x" name="X" />
                        <YAxis type="number" dataKey="y" name="Y" />
                        {(() => {
                            const maxCluster = Math.max(...data.map(d => d.clust));
                            const numClusters = maxCluster + 1;  // Since clusters start from 0

                            return Array.from(new Set(data.map(d => d.clust))).map((cluster) => {
                                const hue = (360 / numClusters) * cluster;
                                return (
                                    <Scatter
                                        key={cluster}
                                        data={data.filter(d => d.clust === cluster)}
                                        fill={`hsl(${hue}deg, 70%, 50%)`}
                                        onClick={(point) => {
                                            const p = point as unknown as DataPoint;
                                            handleClick(p);
                                            handlePlay(p.filename);
                                        }}
                                        cursor="pointer"
                                    />
                                );
                            });
                        })()}
                    </ScatterChart>
                )}
            </div>

            <div className="w-96 border rounded-lg p-4 flex flex-col">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-lg font-semibold">Files</h2>
                    <button
                        onClick={handleReset}
                        className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded"
                    >
                        Reset
                    </button>
                </div>
                <div className="flex-1 overflow-auto">
                    {displayFiles.map((file, index) => (
                        <div
                            key={index}
                            className={`p-2 text-sm ${file.filename === selectedFile ? 'bg-blue-100' : 'hover:bg-gray-50'
                                }`}
                        >
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={() => handlePlay(file.filename)}
                                    className="p-1 hover:bg-gray-200 rounded"
                                >
                                    {currentAudio?.file === file.filename && currentAudio.isPlaying ?
                                        <Pause className="w-4 h-4" /> :
                                        <Play className="w-4 h-4" />
                                    }
                                </button>
                                <span>{file.filename.split('/').pop()}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default AudioVisualizer;