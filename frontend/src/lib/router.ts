import { useEffect, useState } from "react";

export type Route =
  | { name: "tasks" }
  | { name: "skills" }
  | { name: "skill-detail"; skillId: string }
  | { name: "evals" }
  | { name: "architecture" };

export function parseHash(hash: string): Route {
  // Strip leading "#" and optional "/"
  const path = hash.replace(/^#\/?/, "");
  if (path === "" || path === "tasks") return { name: "tasks" };
  if (path === "skills") return { name: "skills" };
  if (path === "evals") return { name: "evals" };
  if (path === "architecture") return { name: "architecture" };
  const m = path.match(/^skills\/([^/]+)$/);
  if (m) return { name: "skill-detail", skillId: m[1] };
  // Unknown hash → treat as tasks (home)
  return { name: "tasks" };
}

export function useHashRoute(): Route {
  const [route, setRoute] = useState<Route>(() =>
    parseHash(window.location.hash),
  );
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}
