"use client";

import React, { useEffect, useState } from "react";
import { Clock, Play, Pause, FastForward, RotateCcw, AlertCircle } from "lucide-react";

interface MemoryEvent {
  id: number;
  timestamp: string;
  event_type: string;
  description: string;
  metadata: any;
}

interface TimelineConsoleProps {
  onTimeTravel: (timestamp: string | null) => void;
  refreshTrigger: number;
}

export default function TimelineConsole({ onTimeTravel, refreshTrigger }: TimelineConsoleProps) {
  const [events, setEvents] = useState<MemoryEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Playback state
  const [sliderIndex, setSliderIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1500); // ms per step

  const fetchEvents = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch((process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000") + "/api/timeline?limit=100");
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to fetch events.");
      
      // Events are returned DESC, reverse them for chronological playback
      const chronEvents = (data.events || []).reverse();
      setEvents(chronEvents);
      setSliderIndex(chronEvents.length > 0 ? chronEvents.length - 1 : 0);
    } catch (err: any) {
      setError(err.message || "Failed to load events timeline.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvents();
  }, [refreshTrigger]);

  // Handle Playback Loop
  useEffect(() => {
    let intervalId: any;
    if (isPlaying && events.length > 0) {
      intervalId = setInterval(() => {
        setSliderIndex((prev) => {
          if (prev >= events.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          const next = prev + 1;
          const targetEvent = events[next];
          onTimeTravel(targetEvent.timestamp);
          return next;
        });
      }, playbackSpeed);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isPlaying, events, playbackSpeed]);

  const debounceTimerRef = React.useRef<any>(null);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const idx = parseInt(e.target.value);
    setSliderIndex(idx);
    setIsPlaying(false);

    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);

    debounceTimerRef.current = setTimeout(() => {
      if (events.length > 0 && idx < events.length) {
        if (idx === events.length - 1) {
          onTimeTravel(null);
        } else {
          onTimeTravel(events[idx].timestamp);
        }
      }
    }, 400);
  };

  const handleReset = () => {
    setIsPlaying(false);
    if (events.length > 0) {
      setSliderIndex(events.length - 1);
      onTimeTravel(null);
    }
  };

  const formatTimestamp = (isoStr: string) => {
    try {
      const dt = new Date(isoStr);
      return dt.toLocaleTimeString() + " " + dt.toLocaleDateString(undefined, { month: "short", day: "numeric" });
    } catch (e) {
      return isoStr;
    }
  };

  const getEventBadgeClass = (type: string) => {
    switch (type) {
      case "MemoryCreated":
        return "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
      case "RecallTriggered":
        return "bg-purple-500/10 text-purple-400 border-purple-500/20";
      case "MemoryImproved":
        return "bg-blue-500/10 text-blue-400 border-blue-500/20";
      case "MemoryForgotten":
        return "bg-slate-500/10 text-slate-400 border-slate-500/20";
      case "AgentAction":
        return "bg-amber-500/10 text-amber-400 border-amber-500/20";
      default:
        return "bg-slate-900 border-slate-800 text-slate-500";
    }
  };

  const activeEvent = events.length > 0 && sliderIndex < events.length ? events[sliderIndex] : null;

  return (
    <div className="bg-slate-900/40 backdrop-blur-md border border-slate-800 p-6 rounded-2xl shadow-xl flex flex-col h-full text-slate-200">
      <h2 className="text-lg font-bold text-white font-outfit mb-1">Memory Time Machine</h2>
      <p className="text-slate-500 text-xs mb-5">
        Drag the timeline slider to rewinding memory snapshots, or play back the knowledge graph's growth step-by-step.
      </p>

      {/* Time Machine Playback Controls */}
      <div className="bg-slate-950/60 border border-slate-800/80 p-4 rounded-xl space-y-4 mb-5 shadow">
        <div className="flex flex-col sm:flex-row items-center gap-4 justify-between">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              disabled={events.length <= 1}
              className={`p-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-bold transition disabled:opacity-40`}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </button>
            <button
              onClick={handleReset}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
              title="Reset to Present"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
            
            {/* Speed Toggle */}
            <select
              value={playbackSpeed}
              onChange={(e) => setPlaybackSpeed(parseInt(e.target.value))}
              className="bg-slate-900 border border-slate-800 rounded-lg text-[10px] px-2 py-1 outline-none font-mono"
            >
              <option value={2500}>0.5x Speed</option>
              <option value={1500}>1.0x Speed</option>
              <option value={800}>2.0x Speed</option>
            </select>
          </div>

          <div className="text-xs font-mono text-slate-400">
            {activeEvent ? (
              <span className="flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 text-blue-400" />
                Playback: {formatTimestamp(activeEvent.timestamp)}
              </span>
            ) : (
              "No logs available"
            )}
          </div>
        </div>

        {/* Timeline Slider */}
        <div className="space-y-1">
          <input
            type="range"
            min={0}
            max={events.length > 0 ? events.length - 1 : 0}
            value={sliderIndex}
            onChange={handleSliderChange}
            disabled={events.length <= 1}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500 disabled:opacity-40"
          />
          <div className="flex justify-between text-[8px] font-mono text-slate-600 uppercase tracking-widest font-bold px-1">
            <span>Genesis Memory</span>
            <span>{sliderIndex === events.length - 1 ? "Present (Latest)" : "Rewound Snapshot"}</span>
          </div>
        </div>
      </div>

      {/* Events Log List */}
      <div className="flex-1 border border-slate-800/80 rounded-xl p-3 bg-slate-950/20 overflow-y-auto max-h-[220px]">
        <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold font-mono border-b border-slate-900 pb-2 mb-2">
          Memory Event Stream
        </div>
        
        {events.length === 0 ? (
          <div className="text-center py-10 text-xs text-slate-600 italic">No events logged yet.</div>
        ) : (
          <div className="space-y-2">
            {/* Show reversed array for DESC presentation in lists */}
            {[...events].reverse().map((ev) => {
              const evIdx = events.findIndex(e => e.id === ev.id);
              const isSelectedInTimeMachine = sliderIndex === evIdx;
              return (
                <div
                  key={ev.id}
                  className={`p-2.5 rounded-lg border text-xs flex justify-between gap-3 items-start transition ${
                    isSelectedInTimeMachine
                      ? "bg-blue-950/15 border-blue-500/30 text-slate-200"
                      : "bg-slate-950/30 border-slate-900 text-slate-400 hover:bg-slate-900/20"
                  }`}
                >
                  <div className="space-y-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`px-1.5 py-0.5 rounded text-[8px] border font-mono ${getEventBadgeClass(ev.event_type)}`}>
                        {ev.event_type}
                      </span>
                      <span className="text-[9px] text-slate-500 font-mono">
                        {formatTimestamp(ev.timestamp)}
                      </span>
                    </div>
                    <div className="font-semibold text-slate-350 truncate">{ev.description}</div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {error && (
        <div className="mt-4 p-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs rounded-xl flex items-center gap-2">
          <AlertCircle className="w-4 h-4" />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
