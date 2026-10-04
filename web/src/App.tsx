import { MotionConfig } from "framer-motion";
import { Route, Routes } from "react-router-dom";
import LandingScreen from "./screens/LandingScreen";
import MapScreen from "./screens/MapScreen";
import PlanetPathScreen from "./screens/PlanetPathScreen";
import MissionScreen from "./screens/MissionScreen";
import DiagnosisScreen from "./screens/DiagnosisScreen";
import ProbeScreen from "./screens/ProbeScreen";
import InterventionScreen from "./screens/InterventionScreen";
import TransferTrapScreen from "./screens/TransferTrapScreen";
import VerdictScreen from "./screens/VerdictScreen";

export default function App() {
  return (
    <MotionConfig reducedMotion="user">
      <Routes>
        <Route path="/" element={<LandingScreen />} />
        <Route path="/map" element={<MapScreen />} />
        <Route path="/planet/conditions" element={<PlanetPathScreen />} />
        <Route path="/planet/conditions/mission/:problemId" element={<MissionScreen />} />
        <Route path="/planet/conditions/diagnosis" element={<DiagnosisScreen />} />
        <Route path="/planet/conditions/probe" element={<ProbeScreen />} />
        <Route path="/planet/conditions/intervention" element={<InterventionScreen />} />
        <Route path="/planet/conditions/trap" element={<TransferTrapScreen />} />
        <Route path="/planet/conditions/verdict" element={<VerdictScreen />} />
      </Routes>
    </MotionConfig>
  );
}
