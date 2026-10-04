import { useNavigate } from "react-router-dom";
import ArtStage from "../components/ArtStage";
import PlanetHotspot from "../components/PlanetHotspot";
import { PLANETS } from "../fixtures/planets";

const PLANET_ROUTES: Record<string, string> = {
  "aegis-grid": "/planet/conditions",
  "miners-belt": "/planet/loops",
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
    </ArtStage>
  );
}
