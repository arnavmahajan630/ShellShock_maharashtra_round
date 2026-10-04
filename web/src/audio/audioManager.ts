/**
 * Audio Manager for Re:Learn.
 * Manages background music tracks, smooth cross-fading, user mute state,
 * and browser autoplay policy unlock.
 */

export type AudioTrackId = "galaxy" | "blackhole";

const TRACK_SOURCES: Record<AudioTrackId, string> = {
  galaxy: "/audio/map-loop.mp3",
  blackhole: "/audio/blackhole.mp3",
};

const STORAGE_KEY_MUTED = "relearn_audio_muted";
const DEFAULT_VOLUME = 0.35;

class AudioManager {
  private audios: Map<AudioTrackId, HTMLAudioElement> = new Map();
  private targetTrack: AudioTrackId | null = null;
  private currentTrack: AudioTrackId | null = null;
  private isMuted: boolean = false;
  private volume: number = DEFAULT_VOLUME;
  private listeners: Set<() => void> = new Set();
  private isUnlocked: boolean = false;
  private fadeInterval: number | null = null;

  constructor() {
    // Read saved preference (default unmuted)
    const saved = localStorage.getItem(STORAGE_KEY_MUTED);
    this.isMuted = saved === "true";

    // Initialize audio elements if in browser
    if (typeof window !== "undefined") {
      for (const [id, src] of Object.entries(TRACK_SOURCES) as [AudioTrackId, string][]) {
        const audio = new Audio(src);
        audio.loop = true;
        audio.preload = "auto";
        audio.volume = 0;
        this.audios.set(id, audio);
      }

      // One-time interaction listener to satisfy browser autoplay restrictions
      const unlock = () => {
        if (this.isUnlocked) return;
        this.isUnlocked = true;
        window.removeEventListener("pointerdown", unlock);
        window.removeEventListener("keydown", unlock);
        if (this.targetTrack && !this.isMuted) {
          this.applyPlayback();
        }
      };

      window.addEventListener("pointerdown", unlock, { passive: true });
      window.addEventListener("keydown", unlock, { passive: true });
    }
  }

  public subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify() {
    this.listeners.forEach((fn) => fn());
  }

  public getActiveTrack(): AudioTrackId | null {
    return this.targetTrack;
  }

  public getIsMuted(): boolean {
    return this.isMuted;
  }

  public getIsPlaying(): boolean {
    if (this.isMuted || !this.currentTrack) return false;
    const audio = this.audios.get(this.currentTrack);
    return audio ? !audio.paused && audio.volume > 0 : false;
  }

  public getVolume(): number {
    return this.volume;
  }

  public setVolume(vol: number) {
    this.volume = Math.max(0, Math.min(1, vol));
    if (this.currentTrack && !this.isMuted) {
      const audio = this.audios.get(this.currentTrack);
      if (audio) audio.volume = this.volume;
    }
    this.notify();
  }

  public setMuted(muted: boolean) {
    if (this.isMuted === muted) return;
    this.isMuted = muted;
    try {
      localStorage.setItem(STORAGE_KEY_MUTED, String(muted));
    } catch {
      // Ignore localStorage errors
    }
    this.applyPlayback();
    this.notify();
  }

  public toggleMute() {
    this.setMuted(!this.isMuted);
  }

  public setTrack(track: AudioTrackId | null) {
    if (this.targetTrack === track) return;
    this.targetTrack = track;
    this.applyPlayback();
    this.notify();
  }

  /**
   * Applies the current targetTrack and isMuted state with smooth cross-fade.
   */
  private applyPlayback() {
    if (this.fadeInterval !== null) {
      window.clearInterval(this.fadeInterval);
      this.fadeInterval = null;
    }

    // If muted or target is null, fade out all playing audios
    if (this.isMuted || !this.targetTrack) {
      this.fadeOutAll();
      this.currentTrack = null;
      return;
    }

    const targetAudio = this.audios.get(this.targetTrack);
    if (!targetAudio) return;

    // If switching from another track, cross-fade
    const previousTrack = this.currentTrack;
    this.currentTrack = this.targetTrack;

    if (previousTrack && previousTrack !== this.targetTrack) {
      const prevAudio = this.audios.get(previousTrack);
      this.crossFade(prevAudio, targetAudio);
    } else {
      // Fade in targetAudio
      if (targetAudio.paused) {
        targetAudio.volume = 0;
        const playPromise = targetAudio.play();
        if (playPromise) {
          playPromise.catch(() => {
            // Autoplay blocked; will unlock on first user gesture
          });
        }
      }
      this.fadeIn(targetAudio);
    }
  }

  private fadeOutAll() {
    const steps = 10;
    const stepTime = 30; // 300ms total
    let currentStep = 0;

    const activeAudios = Array.from(this.audios.values()).filter((a) => !a.paused && a.volume > 0);
    if (activeAudios.length === 0) return;

    const initialVolumes = activeAudios.map((a) => a.volume);

    this.fadeInterval = window.setInterval(() => {
      currentStep++;
      const factor = 1 - currentStep / steps;

      activeAudios.forEach((audio, idx) => {
        audio.volume = Math.max(0, initialVolumes[idx] * factor);
      });

      if (currentStep >= steps) {
        if (this.fadeInterval !== null) {
          window.clearInterval(this.fadeInterval);
          this.fadeInterval = null;
        }
        activeAudios.forEach((audio) => {
          audio.pause();
          audio.volume = 0;
        });
        this.notify();
      }
    }, stepTime);
  }

  private fadeIn(target: HTMLAudioElement) {
    const steps = 10;
    const stepTime = 30;
    let currentStep = 0;
    const targetVol = this.volume;

    this.fadeInterval = window.setInterval(() => {
      currentStep++;
      target.volume = Math.min(targetVol, (currentStep / steps) * targetVol);

      if (currentStep >= steps) {
        if (this.fadeInterval !== null) {
          window.clearInterval(this.fadeInterval);
          this.fadeInterval = null;
        }
        target.volume = targetVol;
        this.notify();
      }
    }, stepTime);
  }

  private crossFade(prevAudio: HTMLAudioElement | undefined, nextAudio: HTMLAudioElement) {
    if (nextAudio.paused) {
      nextAudio.volume = 0;
      const playPromise = nextAudio.play();
      if (playPromise) {
        playPromise.catch(() => {
          // Autoplay blocked
        });
      }
    }

    const steps = 12;
    const stepTime = 25; // 300ms total
    let currentStep = 0;
    const startPrevVol = prevAudio ? prevAudio.volume : 0;
    const targetNextVol = this.volume;

    this.fadeInterval = window.setInterval(() => {
      currentStep++;
      const progress = currentStep / steps;

      if (prevAudio) {
        prevAudio.volume = Math.max(0, startPrevVol * (1 - progress));
      }
      nextAudio.volume = Math.min(targetNextVol, progress * targetNextVol);

      if (currentStep >= steps) {
        if (this.fadeInterval !== null) {
          window.clearInterval(this.fadeInterval);
          this.fadeInterval = null;
        }
        if (prevAudio) {
          prevAudio.pause();
          prevAudio.volume = 0;
        }
        nextAudio.volume = targetNextVol;
        this.notify();
      }
    }, stepTime);
  }
}

export const audioManager = new AudioManager();
