'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  AlertCircle, 
  Camera, 
  CameraOff, 
  Brain, 
  Activity, 
  Heart,
  Eye,
  Zap,
  Target,
  TrendingUp,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  Info
} from 'lucide-react';

// ── Types ────────────────────────────────────────────────────────────────────

interface EmotionProbs {
  angry: number;
  disgust: number;
  fear: number;
  happy: number;
  sad: number;
  surprise: number;
  neutral: number;
}

interface AnalysisResult {
  success: boolean;
  face_detected: boolean;
  emotion_probs: EmotionProbs;
  dominant_emotion: string;
  valence: number;
  arousal: number;
  stress: number;
  fatigue: number;
  attention: number;
  engagement: number;
  mood_label: string;
  mood_quadrant: string;
  blink_rate: number;
  ear_avg: number;
  timestamp: number;
  message: string;
}

interface HistoryData {
  timestamps: number[];
  metrics: {
    valence: number[];
    arousal: number[];
    stress: number[];
    fatigue: number[];
    attention: number[];
    engagement: number[];
  };
  emotion_distribution: Record<string, number>;
}

interface ModelStatus {
  trained: boolean;
  message: string;
  instructions?: string[];
}

// ── Constants ────────────────────────────────────────────────────────────────

const EMOTION_COLORS: Record<string, string> = {
  angry: '#ff4757',
  disgust: '#a55eea',
  fear: '#ff6b81',
  happy: '#2ed573',
  sad: '#70a1ff',
  surprise: '#ffa502',
  neutral: '#dfe6e9',
};

const EMOTION_EMOJIS: Record<string, string> = {
  angry: '😠',
  disgust: '🤢',
  fear: '😨',
  happy: '😊',
  sad: '😢',
  surprise: '😲',
  neutral: '😐',
};

const ML_SERVICE_PORT = 5000;

// ── Utility Functions ────────────────────────────────────────────────────────

function getMetricColor(value: number, inverse: boolean = false): string {
  const v = inverse ? 1 - value : value;
  if (v >= 0.66) return '#2ed573';
  if (v >= 0.33) return '#ffa502';
  return '#ff4757';
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

// ── Components ────────────────────────────────────────────────────────────────

function MetricCard({ 
  title, 
  value, 
  icon: Icon, 
  inverse = false,
  description 
}: { 
  title: string; 
  value: number; 
  icon: React.ElementType;
  inverse?: boolean;
  description: string;
}) {
  const color = getMetricColor(value, inverse);
  
  return (
    <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
      <CardContent className="p-4">
        <div className="flex items-center gap-2 mb-2">
          <Icon className="w-4 h-4 text-white/60" />
          <span className="text-xs text-white/60 uppercase tracking-wider">{title}</span>
        </div>
        <div className="text-3xl font-light" style={{ color }}>
          {formatPercent(value)}
        </div>
        <Progress 
          value={value * 100} 
          className="h-1 mt-2 bg-white/10"
          style={{ 
            // @ts-ignore
            '--progress-background': color 
          }}
        />
        <p className="text-xs text-white/40 mt-1">{description}</p>
      </CardContent>
    </Card>
  );
}

function EmotionRadar({ probs }: { probs: EmotionProbs }) {
  const emotions = Object.keys(probs);
  const values = Object.values(probs);
  
  // SVG radar chart dimensions
  const size = 200;
  const center = size / 2;
  const radius = 80;
  const angleStep = (2 * Math.PI) / emotions.length;
  
  // Calculate points
  const points = emotions.map((_, i) => {
    const angle = i * angleStep - Math.PI / 2;
    return {
      x: center + radius * Math.cos(angle),
      y: center + radius * Math.sin(angle),
    };
  });
  
  // Data points
  const dataPoints = emotions.map((emotion, i) => {
    const angle = i * angleStep - Math.PI / 2;
    const r = radius * values[i];
    return {
      x: center + r * Math.cos(angle),
      y: center + r * Math.sin(angle),
      emotion,
      value: values[i],
    };
  });
  
  const pathData = dataPoints.map((p, i) => 
    `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`
  ).join(' ') + ' Z';

  return (
    <div className="relative">
      <svg width={size} height={size} className="mx-auto">
        {/* Background circles */}
        {[0.25, 0.5, 0.75, 1].map((r, i) => (
          <circle
            key={i}
            cx={center}
            cy={center}
            r={radius * r}
            fill="none"
            stroke="rgba(255,255,255,0.1)"
            strokeWidth={1}
          />
        ))}
        
        {/* Axis lines */}
        {emotions.map((_, i) => {
          const angle = i * angleStep - Math.PI / 2;
          return (
            <line
              key={i}
              x1={center}
              y1={center}
              x2={center + radius * Math.cos(angle)}
              y2={center + radius * Math.sin(angle)}
              stroke="rgba(255,255,255,0.1)"
              strokeWidth={1}
            />
          );
        })}
        
        {/* Data polygon */}
        <path
          d={pathData}
          fill="rgba(0, 255, 150, 0.15)"
          stroke="rgba(0, 255, 150, 0.8)"
          strokeWidth={2}
        />
        
        {/* Data points */}
        {dataPoints.map((p, i) => (
          <circle
            key={i}
            cx={p.x}
            cy={p.y}
            r={4}
            fill={EMOTION_COLORS[p.emotion]}
            stroke="white"
            strokeWidth={1}
          />
        ))}
        
        {/* Labels */}
        {emotions.map((emotion, i) => {
          const angle = i * angleStep - Math.PI / 2;
          const labelRadius = radius + 20;
          const x = center + labelRadius * Math.cos(angle);
          const y = center + labelRadius * Math.sin(angle);
          
          return (
            <text
              key={emotion}
              x={x}
              y={y}
              textAnchor="middle"
              dominantBaseline="middle"
              className="text-xs fill-white/60"
            >
              {EMOTION_EMOJIS[emotion]}
            </text>
          );
        })}
      </svg>
      
      {/* Legend */}
      <div className="flex flex-wrap justify-center gap-2 mt-2">
        {emotions.map((emotion) => (
          <div key={emotion} className="flex items-center gap-1 text-xs text-white/60">
            <div 
              className="w-2 h-2 rounded-full" 
              style={{ backgroundColor: EMOTION_COLORS[emotion] }}
            />
            <span>{emotion}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TimelineChart({ 
  data, 
  color = '#ff6655',
  title 
}: { 
  data: number[]; 
  color?: string;
  title: string;
}) {
  if (data.length === 0) {
    return (
      <div className="h-24 flex items-center justify-center text-white/40 text-sm">
        No data yet
      </div>
    );
  }
  
  const height = 80;
  const width = 100;
  const points = data.map((v, i) => {
    const x = (i / Math.max(data.length - 1, 1)) * width;
    const y = height - v * height;
    return `${x},${y}`;
  }).join(' ');
  
  // Create area path
  const areaPoints = `0,${height} ${points} ${width},${height}`;
  
  return (
    <div>
      <div className="text-xs text-white/60 mb-1">{title}</div>
      <svg 
        viewBox={`0 0 ${width} ${height}`} 
        className="w-full h-24"
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id={`gradient-${color}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.3" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        
        {/* Area fill */}
        <polygon
          points={areaPoints}
          fill={`url(#gradient-${color})`}
        />
        
        {/* Line */}
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

function StatusIndicator({ status }: { status: ModelStatus }) {
  return (
    <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          {status.trained ? (
            <CheckCircle className="w-5 h-5 text-green-400" />
          ) : (
            <AlertTriangle className="w-5 h-5 text-yellow-400" />
          )}
          <div>
            <div className="text-sm font-medium">
              {status.trained ? 'Model Ready' : 'Model Not Trained'}
            </div>
            <div className="text-xs text-white/60">{status.message}</div>
          </div>
        </div>
        
        {status.instructions && (
          <div className="mt-3 p-3 bg-white/5 rounded-lg">
            <div className="text-xs text-white/40 mb-2">Setup Instructions:</div>
            {status.instructions.map((instruction, i) => (
              <div key={i} className="text-xs text-white/60 font-mono">
                {instruction}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────

export default function VibeCheckDashboard() {
  // State
  const [isStreaming, setIsStreaming] = useState(false);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [history, setHistory] = useState<HistoryData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  
  // Refs
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number | null>(null);
  const lastFrameTime = useRef<number>(0);
  const frameCount = useRef<number>(0);
  const fpsUpdateTime = useRef<number>(0);
  
  // Initialize - check model status once on mount
  useEffect(() => {
    let mounted = true;
    
    const initModelStatus = async () => {
      try {
        const response = await fetch('/api/ml?path=model/status');
        const data = await response.json();
        if (mounted) {
          setModelStatus(data);
        }
      } catch {
        if (mounted) {
          setModelStatus({
            trained: false,
            message: 'Cannot connect to ML service',
            instructions: [
              '1. Navigate to vibe-check directory',
              '2. Run: uvicorn app.api:app --host 0.0.0.0 --port 5000',
            ],
          });
        }
      }
    };
    
    initModelStatus();
    
    return () => {
      mounted = false;
    };
  }, []);
  
  // Periodically check history when streaming
  useEffect(() => {
    if (!isStreaming) return;
    
    const fetchHistory = async () => {
      try {
        const response = await fetch('/api/ml?path=history');
        const data = await response.json();
        setHistory(data);
      } catch (err) {
        console.error('Failed to fetch history:', err);
      }
    };
    
    fetchHistory();
    const interval = setInterval(fetchHistory, 1000);
    
    return () => clearInterval(interval);
  }, [isStreaming]);
  
  // Start camera
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      
      streamRef.current = stream;
      setIsStreaming(true);
      setError(null);
    } catch (err) {
      console.error('Camera error:', err);
      setError('Could not access camera. Please grant camera permissions.');
    }
  }, []);
  
  // Stop camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    setIsStreaming(false);
    setResult(null);
  }, []);
  
  // Ref for the analysis function to avoid circular dependency
  const analyzeFrameRef = useRef<(() => void) | null>(null);
  
  // Capture and analyze frame
  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx || video.readyState !== video.HAVE_ENOUGH_DATA) {
      if (analyzeFrameRef.current) {
        animationRef.current = requestAnimationFrame(analyzeFrameRef.current);
      }
      return;
    }
    
    // Calculate FPS
    const now = performance.now();
    frameCount.current++;
    
    if (now - fpsUpdateTime.current >= 1000) {
      setFps(frameCount.current);
      frameCount.current = 0;
      fpsUpdateTime.current = now;
    }
    
    // Throttle to ~10 FPS for analysis (reduce server load)
    if (now - lastFrameTime.current < 100) {
      if (analyzeFrameRef.current) {
        animationRef.current = requestAnimationFrame(analyzeFrameRef.current);
      }
      return;
    }
    lastFrameTime.current = now;
    
    // Draw frame to canvas
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Mirror for selfie mode
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    
    // Convert to base64
    const imageBase64 = canvas.toDataURL('image/jpeg', 0.8);
    
    // Send to ML service
    try {
      const response = await fetch('/api/ml?path=analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: imageBase64.split(',')[1],
          timestamp: Date.now() / 1000,
        }),
      });
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('Analysis error:', err);
    }
    
    // Continue loop
    if (analyzeFrameRef.current) {
      animationRef.current = requestAnimationFrame(analyzeFrameRef.current);
    }
  }, []);
  
  // Update the ref when the function changes
  useEffect(() => {
    analyzeFrameRef.current = captureAndAnalyze;
  }, [captureAndAnalyze]);
  
  // Start analysis loop when streaming
  useEffect(() => {
    if (isStreaming && modelStatus?.trained) {
      animationRef.current = requestAnimationFrame(captureAndAnalyze);
    }
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isStreaming, modelStatus?.trained, captureAndAnalyze]);
  
  return (
    <div className="min-h-screen bg-[#07070f] text-white">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-white/10 bg-black/60 backdrop-blur-xl">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <svg width="42" height="42" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ filter: 'drop-shadow(0px 0px 8px rgba(88, 166, 255, 0.5))' }}>
                <defs>
                  <linearGradient id="primaryGrad" x1="0" y1="0" x2="48" y2="48">
                    <stop stopColor="#a371f7" />
                    <stop offset="1" stopColor="#58a6ff" />
                  </linearGradient>
                  <linearGradient id="secondaryGrad" x1="48" y1="0" x2="0" y2="48">
                    <stop stopColor="#58a6ff" />
                    <stop offset="1" stopColor="rgba(88, 166, 255, 0.2)" />
                  </linearGradient>
                </defs>
                <path d="M12 14 L24 38 L36 14" stroke="url(#primaryGrad)" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M18 24 L26 32 L42 8" stroke="url(#secondaryGrad)" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
                <circle cx="24" cy="38" r="4" fill="#ffffff" />
              </svg>
              <div>
                <h1 className="text-2xl font-bold tracking-wider m-0 leading-tight">
                  <span className="bg-gradient-to-r from-[#58a6ff] to-[#a371f7] bg-clip-text text-transparent">Vibe</span>
                  <span className="text-white font-light">Check</span>
                </h1>
                <p className="text-[10px] text-[#58a6ff]/80 uppercase tracking-[0.2em] font-medium mt-1">AI Emotion Engine</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              {fps > 0 && (
                <Badge variant="outline" className="border-white/20">
                  {fps} FPS
                </Badge>
              )}
              
              {modelStatus?.trained ? (
                <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                  Model Ready
                </Badge>
              ) : (
                <Badge className="bg-yellow-500/20 text-yellow-400 border-yellow-500/30">
                  Model Not Trained
                </Badge>
              )}
              
              <Button
                onClick={isStreaming ? stopCamera : startCamera}
                variant={isStreaming ? "destructive" : "default"}
                className="gap-2"
              >
                {isStreaming ? (
                  <>
                    <CameraOff className="w-4 h-4" />
                    Stop
                  </>
                ) : (
                  <>
                    <Camera className="w-4 h-4" />
                    Start Camera
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        {error && (
          <div className="mb-6 p-4 bg-red-500/20 border border-red-500/30 rounded-lg flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <span className="text-red-200">{error}</span>
          </div>
        )}
        
        {/* Model Status Warning */}
        {modelStatus && !modelStatus.trained && (
          <div className="mb-6">
            <StatusIndicator status={modelStatus} />
          </div>
        )}
        
        {/* Dashboard Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Camera */}
          <div className="lg:col-span-2 space-y-6">
            {/* Camera Feed */}
            <Card className="bg-black/40 border-white/10 backdrop-blur-sm overflow-hidden">
              <CardContent className="p-0">
                <div className="relative aspect-video bg-black">
                  <video
                    ref={videoRef}
                    className={`absolute inset-0 w-full h-full object-cover ${isStreaming ? '' : 'hidden'}`}
                    playsInline
                    muted
                  />
                  
                  {!isStreaming && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-white/40">
                      <Camera className="w-16 h-16 mb-4" />
                      <p>Click "Start Camera" to begin</p>
                    </div>
                  )}
                  
                  {/* Overlay with mood */}
                  {isStreaming && result?.face_detected && (
                    <div className="absolute top-4 left-4">
                      <div className="bg-black/60 backdrop-blur-sm rounded-lg px-4 py-2">
                        <div className="text-2xl font-light text-emerald-400">
                          {result.mood_label}
                        </div>
                        <div className="text-xs text-white/60">
                          {EMOTION_EMOJIS[result.dominant_emotion]} {result.dominant_emotion}
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {/* No face message */}
                  {isStreaming && result && !result.face_detected && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="bg-black/60 backdrop-blur-sm rounded-lg px-6 py-4 text-white/60">
                        <div className="flex items-center gap-2">
                          <Eye className="w-5 h-5" />
                          <span>No face detected</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
            
            
            {/* Metrics Grid */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <MetricCard
                title="Valence"
                value={((result?.valence ?? 0) + 1) / 2}
                icon={Heart}
                description="😊 ↔ 😢"
              />
              <MetricCard
                title="Arousal"
                value={result?.arousal ?? 0}
                icon={Zap}
                description="⚡ energy"
              />
              <MetricCard
                title="Stress"
                value={result?.stress ?? 0}
                icon={Activity}
                inverse
                description="😰 tension"
              />
              <MetricCard
                title="Fatigue"
                value={result?.fatigue ?? 0}
                icon={Eye}
                inverse
                description="😴 tiredness"
              />
              <MetricCard
                title="Attention"
                value={result?.attention ?? 0}
                icon={Target}
                description="👁 focus"
              />
              <MetricCard
                title="Engagement"
                value={result?.engagement ?? 0}
                icon={TrendingUp}
                description="🎯 interest"
              />
            </div>
          </div>
          
          {/* Right Column - Charts */}
          <div className="space-y-6">
            {/* Emotion Radar */}
            <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-white/80">
                  Emotion Radar
                </CardTitle>
              </CardHeader>
              <CardContent>
                <EmotionRadar probs={result?.emotion_probs || {
                  angry: 0.14, disgust: 0.14, fear: 0.14,
                  happy: 0.14, sad: 0.14, surprise: 0.14, neutral: 0.14
                }} />
              </CardContent>
            </Card>
            
            {/* Timeline */}
            <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-white/80">
                  Timeline (60s)
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <TimelineChart
                  data={history?.metrics.stress || []}
                  color="#ff6655"
                  title="Stress"
                />
                <TimelineChart
                  data={history?.metrics.attention || []}
                  color="#2ed573"
                  title="Attention"
                />
                <TimelineChart
                  data={history?.metrics.engagement || []}
                  color="#ffa502"
                  title="Engagement"
                />
              </CardContent>
            </Card>
            
            {/* Emotion Distribution */}
            {history?.emotion_distribution && (
              <Card className="bg-black/40 border-white/10 backdrop-blur-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-white/80">
                    Emotion Distribution
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {(Object.entries(history.emotion_distribution) as [string, number][])
                      .sort(([, a], [, b]) => b - a)
                      .map(([emotion, count]) => (
                        <div key={emotion} className="flex items-center gap-2">
                          <span className="text-lg">{EMOTION_EMOJIS[emotion]}</span>
                          <span className="text-sm text-white/60 flex-1">{emotion}</span>
                          <Badge variant="outline" className="border-white/20">
                            {count}
                          </Badge>
                        </div>
                      ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </main>
      
      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} className="hidden" />
      
      {/* Footer */}
      <footer className="fixed bottom-0 left-0 right-0 border-t border-white/10 bg-black/60 backdrop-blur-xl">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between text-xs text-white/40">
            <div className="flex items-center gap-4">
              <span>Vibe Check v1.0</span>
              <span>•</span>
              <span>FER2013 CNN • MediaPipe FaceMesh</span>
            </div>
            <div className="flex items-center gap-4">
              {result?.ear_avg !== undefined && (
                <span>EAR: {result.ear_avg.toFixed(2)}</span>
              )}
              {result?.blink_rate !== undefined && (
                <span>Blinks: {result.blink_rate.toFixed(0)}/min</span>
              )}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
