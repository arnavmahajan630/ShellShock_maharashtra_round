/**
 * Speech Recognition and Speech Synthesis utilities for Re:Learn Multimodal Voice levels.
 */

// Hedge lexicon for cognitive confidence analysis
export const HEDGE_WORDS = [
  "i think",
  "maybe",
  "probably",
  "not sure",
  "guess",
  "i dont know",
  "i don't know",
  "might",
  "could be",
  "um",
  "uh",
  "er",
  "ah",
];

export function countHedges(text: string): number {
  if (!text) return 0;
  const lower = text.toLowerCase();
  let count = 0;
  for (const phrase of HEDGE_WORDS) {
    let pos = 0;
    while ((pos = lower.indexOf(phrase, pos)) !== -1) {
      count++;
      pos += phrase.length;
    }
  }
  return count;
}

export function isSpeechRecognitionSupported(): boolean {
  return typeof window !== "undefined" && ("webkitSpeechRecognition" in window || "SpeechRecognition" in window);
}

export interface SpeechListenerOptions {
  onTranscript: (transcript: string, isFinal: boolean) => void;
  onError: (error: string) => void;
  onEnd: () => void;
}

export class SpeechRecorder {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private recognition: any = null;
  private isRecording: boolean = false;
  private transcript: string = "";
  private startTime: number = 0;

  constructor(private options: SpeechListenerOptions) {
    if (isSpeechRecognitionSupported()) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const SpeechRecognitionClass = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      this.recognition = new SpeechRecognitionClass();
      this.recognition.continuous = true;
      this.recognition.interimResults = true;
      this.recognition.lang = "en-US";

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      this.recognition.onresult = (event: any) => {
        let interim = "";
        let final = "";
        for (let i = 0; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            final += event.results[i][0].transcript + " ";
          } else {
            interim += event.results[i][0].transcript;
          }
        }
        const current = (final + interim).trim();
        this.transcript = current;
        this.options.onTranscript(current, false);
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      this.recognition.onerror = (event: any) => {
        this.options.onError(event.error || "Speech recognition error.");
      };

      this.recognition.onend = () => {
        this.isRecording = false;
        this.options.onEnd();
      };
    }
  }

  public start() {
    if (!this.recognition) {
      this.options.onError("Speech recognition not supported in this browser.");
      return;
    }
    this.transcript = "";
    this.startTime = Date.now();
    this.isRecording = true;
    try {
      this.recognition.start();
    } catch {
      // Handle already started state
    }
  }

  public stop(): { transcript: string; durationMs: number; wordsPerSec: number } {
    if (this.recognition && this.isRecording) {
      try {
        this.recognition.stop();
      } catch {
        // Ignore stop error
      }
    }
    this.isRecording = false;
    const durationMs = Math.max(100, Date.now() - this.startTime);
    const words = this.transcript.trim().split(/\s+/).filter(Boolean).length;
    const wordsPerSec = Math.round((words / (durationMs / 1000)) * 10) / 10;
    return {
      transcript: this.transcript,
      durationMs,
      wordsPerSec,
    };
  }
}
