// Optional 3D stays separate from navigation, recordings and robot controls.
let viewerModule,
  renderStarted = false,
  attempt = 0;
async function getViewer() {
  if (!viewerModule) {
    const version = `20261018-connection-${attempt++}`;
    viewerModule = import(`./viewer.js?v=${version}`).catch((error) => {
      viewerModule = null;
      throw error;
    });
  }
  return viewerModule;
}
export function createScenePreview(element, options = {}) {
  const fallback = [...element.childNodes];
  const button = element.querySelector(".scene-load");
  const status = element.querySelector(".scene-status");
  const controls = options.controls || [];
  let state = "idle",
    view,
    generation = 0;
  controls.forEach((control) => {
    control.disabled = true;
  });
  button.disabled = false;
  function unavailable() {
    view?.dispose();
    view = null;
    element.replaceChildren(...fallback);
    element.dataset.sceneState = state = "unavailable";
    status.textContent =
      options.unavailable ||
      "3D is unavailable here. You can still watch the recordings.";
    button.textContent = "Try 3D again";
    button.disabled = false;
    controls.forEach((control) => {
      control.disabled = true;
    });
    options.onError?.();
  }
  async function load() {
    if (state === "loading" || state === "ready") return;
    element.dataset.sceneState = state = "loading";
    const request = ++generation;
    button.disabled = true;
    button.textContent = "Loading 3D…";
    status.textContent =
      "The recorded preview stays visible while the scene loads.";
    try {
      const module = await getViewer();
      const scene = await (options.getScene?.() || module.loadScene());
      if (request !== generation) return;
      view = new module.TwinView(element, scene, options.dark || false);
      element.dataset.sceneState = state = "ready";
      controls.forEach((control) => {
        control.disabled = false;
      });
      view.renderer.domElement.addEventListener(
        "webglcontextlost",
        (event) => {
          event.preventDefault();
          unavailable();
        },
        { once: true },
      );
      options.onReady?.(view, scene);
      if (!renderStarted) {
        renderStarted = true;
        const draw = () => {
          module.drawViews();
          requestAnimationFrame(draw);
        };
        requestAnimationFrame(draw);
      }
    } catch {
      if (request === generation) unavailable();
    }
  }
  button.onclick = load;
  if (options.auto === "visible" && "IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          observer.disconnect();
          load();
        }
      },
      { rootMargin: "180px" },
    );
    observer.observe(element);
  } else if (options.auto) {
    if ("requestIdleCallback" in window)
      requestIdleCallback(load, { timeout: 1200 });
    else setTimeout(load, 0);
  }
  function reload() {
    generation++;
    view?.dispose();
    view = null;
    options.onError?.();
    element.replaceChildren(...fallback);
    state = "idle";
    return load();
  }
  return { load, reload };
}
