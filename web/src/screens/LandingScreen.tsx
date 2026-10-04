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
          scale: 1.04,
          boxShadow: "0 0 32px 8px var(--color-starlight)",
        }}
        whileTap={{ scale: 0.96 }}
        className="absolute -translate-x-1/2 -translate-y-1/2 cursor-pointer rounded-md outline-none focus-visible:ring-4 focus-visible:ring-warp-cyan/70"
        style={{
          top: "65.8%",
          left: "48.8%",
          width: "21.5%",
          height: "6.9%",
        }}
        aria-label="Start Journey"
      />
    </ArtStage>
  );
}
