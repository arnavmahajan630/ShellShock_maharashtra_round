import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import ArtStage from "../components/ArtStage";

export default function TrialsLandingScreen() {
  const navigate = useNavigate();

  return (
    <ArtStage
      src="/images/trials_landing.png"
      width={1672}
      height={941}
      alt="Deep Space Trials — Adaptive DSA Exam Landing"
    >
      {/* Top-Left Back to Galaxy Hitbox */}
      <motion.button
        type="button"
        onClick={() => navigate("/map")}
        whileHover={{ scale: 1.03, boxShadow: "0 0 20px 4px var(--color-warp-cyan)" }}
        whileTap={{ scale: 0.97 }}
        className="absolute cursor-pointer rounded-lg outline-none focus-visible:ring-4 focus-visible:ring-warp-cyan/70"
        style={{
          top: "7.7%",
          left: "4.8%",
          width: "16.1%",
          height: "7.1%",
        }}
        aria-label="Back to Galaxy"
      />

      {/* Primary Glowing BEGIN TRIALS Button Hitbox */}
      <motion.button
        type="button"
        onClick={() => navigate("/trials/exam")}
        whileHover={{
          scale: 1.03,
          boxShadow: "0 0 30px 8px rgba(0, 240, 255, 0.85)",
        }}
        whileTap={{ scale: 0.97 }}
        className="absolute cursor-pointer rounded-xl outline-none focus-visible:ring-4 focus-visible:ring-warp-cyan"
        style={{
          top: "75.8%",
          left: "47.8%",
          width: "23.0%",
          height: "8.0%",
        }}
        aria-label="Begin Trials"
      />
    </ArtStage>
  );
}
