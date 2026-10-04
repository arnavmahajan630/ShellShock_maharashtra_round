import React, { createContext, useContext, useEffect, useState, useMemo } from "react";
import { useLocation } from "react-router-dom";
import { audioManager, type AudioTrackId } from "./audioManager";

interface AudioContextValue {
  isMuted: boolean;
  isPlaying: boolean;
  activeTrack: AudioTrackId | null;
  toggleMute: () => void;
  setMuted: (muted: boolean) => void;
  volume: number;
  setVolume: (v: number) => void;
}

const AudioContext = createContext<AudioContextValue | null>(null);

export function AudioProvider({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const [isMuted, setIsMuted] = useState(audioManager.getIsMuted());
  const [isPlaying, setIsPlaying] = useState(audioManager.getIsPlaying());
  const [volume, setVolumeState] = useState(audioManager.getVolume());

  // Subscribe to audioManager updates
  useEffect(() => {
    return audioManager.subscribe(() => {
      setIsMuted(audioManager.getIsMuted());
      setIsPlaying(audioManager.getIsPlaying());
      setVolumeState(audioManager.getVolume());
    });
  }, []);

  // Determine current route context
  const pathname = location.pathname;
  const isBlackHole = pathname.startsWith("/trials");

  // Route-based track resolution: music plays continuously, switching between blackhole and galaxy
  useEffect(() => {
    const targetTrack: AudioTrackId = isBlackHole ? "blackhole" : "galaxy";
    audioManager.setTrack(targetTrack);
  }, [isBlackHole]);

  const toggleMute = () => {
    audioManager.toggleMute();
  };

  const setMuted = (muted: boolean) => {
    audioManager.setMuted(muted);
  };

  const setVolume = (v: number) => {
    audioManager.setVolume(v);
  };

  const value = useMemo(
    () => ({
      isMuted,
      isPlaying,
      activeTrack: isBlackHole ? ("blackhole" as const) : ("galaxy" as const),
      toggleMute,
      setMuted,
      volume,
      setVolume,
    }),
    [isMuted, isPlaying, isBlackHole, volume]
  );

  return <AudioContext.Provider value={value}>{children}</AudioContext.Provider>;
}

export function useAudio(): AudioContextValue {
  const ctx = useContext(AudioContext);
  if (!ctx) {
    throw new Error("useAudio must be used within an AudioProvider");
  }
  return ctx;
}
