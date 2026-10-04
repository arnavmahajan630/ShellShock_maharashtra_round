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
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.98, y: 1 }}
        className="absolute cursor-pointer rounded-xl outline-none transition-all duration-200 hover:border-2 hover:border-cyan-300 hover:bg-cyan-400/20 hover:backdrop-brightness-125 hover:shadow-[0_0_20px_rgba(0,240,255,0.7),inset_0_0_10px_rgba(255,255,255,0.3)] focus-visible:ring-4 focus-visible:ring-cyan-400/70"
        style={{
          top: "7.86%",
          left: "2.99%",
          width: "20.87%",
          height: "9.03%",
        }}
        aria-label="Back to Galaxy"
      />

      {/* Primary Glowing BEGIN TRIALS Button Hitbox */}
      <motion.button
        type="button"
        onClick={() => navigate("/trials/exam")}
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.98, y: 2 }}
        className="absolute cursor-pointer rounded-2xl outline-none transition-all duration-200 hover:border-2 hover:border-white/90 hover:bg-white/20 hover:backdrop-brightness-125 hover:shadow-[0_0_35px_rgba(0,240,255,0.85),inset_0_0_15px_rgba(255,255,255,0.4)] focus-visible:ring-4 focus-visible:ring-cyan-400"
        style={{
          top: "72.26%",
          left: "47.61%",
          width: "30.02%",
          height: "17.96%",
        }}
        aria-label="Begin Trials"
      />
    </ArtStage>
  );
}
