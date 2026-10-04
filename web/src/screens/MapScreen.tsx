import { useNavigate } from "react-router-dom";
import ArtStage from "../components/ArtStage";
import PlanetHotspot from "../components/PlanetHotspot";
import { PLANETS } from "../fixtures/planets";

const PLANET_ROUTES: Record<string, string> = {
  "aegis-grid": "/planet/conditions",
  "miners-belt": "/planet/loops",
  "lost-fleet": "/planet/arrays",
  "nav-core": "/planet/variables",
  "module-deck": "/planet/functions",
  "deep-space-trials": "/trials/warp",
};

export default function MapScreen() {
  const navigate = useNavigate();

  return (
    <ArtStage src="/images/new_landing.png" width={1536} height={1024} alt="Re:Learn — galaxy map">
      {PLANETS.map((planet, index) => {
        const route = PLANET_ROUTES[planet.id];
        return (
          <PlanetHotspot
            key={planet.id}
            planet={planet}
            index={index}
            onActivate={route ? () => navigate(route) : undefined}
          />
        );
      })}
      <button
        type="button"
        onClick={() => navigate("/chart")}
        className="fixed left-4 bottom-4 z-10 font-ui text-xl px-5 py-2 rounded border border-warp-cyan bg-deep-space/90 text-warp-cyan hover:bg-warp-cyan hover:text-deep-space active:scale-95"
      >
        ✦ STAR CHART
      </button>
    </ArtStage>
  );
}
