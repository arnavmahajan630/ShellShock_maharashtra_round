import { MotionConfig } from "framer-motion";
import { Route, Routes } from "react-router-dom";
import LandingScreen from "./screens/LandingScreen";
import MapScreen from "./screens/MapScreen";

export default function App() {
  return (
    <MotionConfig reducedMotion="user">
      <Routes>
        <Route path="/" element={<LandingScreen />} />
        <Route path="/map" element={<MapScreen />} />
      </Routes>
    </MotionConfig>
  );
}
