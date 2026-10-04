import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import ArtStage from "../components/ArtStage";

export default function LandingScreen() {
  const navigate = useNavigate();

  return (
    <ArtStage src="/images/landing.png" width={1672} height={941} alt="Re:Learn — title screen">
      <motion.button
        type="button"
        onClick={() => navigate("/map")}
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, duration: 0.5, ease: "easeOut" }}
        whileHover={{
          scale: 1.01,
        }}
        whileTap={{ scale: 0.98, y: 2 }}
        className="absolute cursor-pointer rounded-[18px] outline-none transition-all duration-200 hover:border-2 hover:border-white/80 hover:bg-yellow-300/20 hover:backdrop-brightness-125 hover:shadow-[0_0_30px_rgba(251,191,36,0.75),inset_0_0_15px_rgba(255,255,255,0.4)] focus-visible:ring-4 focus-visible:ring-yellow-300/70"
        style={{
          top: "59.62%",
          left: "33.61%",
          width: "29.49%",
          height: "15.20%",
        }}
        aria-label="Start Journey"
      />
    </ArtStage>
  );
}
