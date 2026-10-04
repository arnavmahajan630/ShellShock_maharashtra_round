import ArtStage from "../components/ArtStage";
import PlanetHotspot from "../components/PlanetHotspot";
import { PLANETS } from "../fixtures/planets";

export default function MapScreen() {
  return (
    <ArtStage src="/images/map.png" width={1536} height={1024} alt="Re:Learn — galaxy map">
      {PLANETS.map((planet, index) => (
        <PlanetHotspot key={planet.id} planet={planet} index={index} />
      ))}
    </ArtStage>
  );
}
