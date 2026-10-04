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
        <Route path="/planet/:planet" element={<PlanetPathScreen />} />
        <Route path="/planet/:planet/mission/:problemId" element={<MissionScreen />} />
        <Route path="/planet/:planet/diagnosis" element={<DiagnosisScreen />} />
        <Route path="/planet/:planet/probe" element={<ProbeScreen />} />
        <Route path="/planet/:planet/intervention" element={<InterventionScreen />} />
        <Route path="/planet/:planet/trap" element={<TransferTrapScreen />} />
        <Route path="/planet/:planet/verdict" element={<VerdictScreen />} />
      </Routes>
    </MotionConfig>
  );
}
