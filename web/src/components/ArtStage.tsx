import type { PropsWithChildren } from "react";

interface ArtStageProps {
  src: string;
  /** Natural pixel dimensions of the art, used to lock the aspect ratio so % overlay coords stay aligned. */
  width: number;
  height: number;
  alt: string;
}

/**
 * Centers a full-bleed background illustration at its native aspect ratio and lets
 * children be positioned over it with percentage-based top/left coordinates that
 * always line up with the art, regardless of viewport width.
 */
export default function ArtStage({
  src,
  width,
  height,
  alt,
  children,
}: PropsWithChildren<ArtStageProps>) {
  return (
    <div className="flex min-h-screen w-full items-center justify-center bg-deep-space">
      <div
        role="img"
        aria-label={alt}
        className="relative w-full bg-cover bg-center"
        style={{
          aspectRatio: `${width} / ${height}`,
          backgroundImage: `url(${src})`,
        }}
      >
        {children}
      </div>
    </div>
  );
}
