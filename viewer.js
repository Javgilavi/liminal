import * as T from "three";
import { OrbitControls } from "./assets/vendor/OrbitControls.js";
const views=[], meshCache=new Map();
const quaternion=a=>new T.Quaternion(a[1],a[2],a[3],a[0]);
const unpack=(b,A)=>new A(Uint8Array.from(atob(b),c=>c.charCodeAt(0)).buffer);
export class TwinView {
  constructor(element, data, dark = false) {
    this.element = element;
    this.scene = new T.Scene();
    this.scene.background = new T.Color(dark ? "#1c3022" : "#edf1e7");
    this.camera = new T.PerspectiveCamera(34, 1, 0.01, 30);
    this.camera.up.set(0, 0, 1);
    this.renderer = new T.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.6));
    this.renderer.outputColorSpace = T.SRGBColorSpace;
    this.renderer.toneMapping = T.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = dark ? 1.2 : 1.05;
    this.renderer.shadowMap.enabled = !dark;
    this.renderer.shadowMap.type = T.PCFSoftShadowMap;
    element.replaceChildren(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = !matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    this.dirty = true;
    this.controls.addEventListener("change", () => {
      this.dirty = true;
    });
    this.controls.minDistance = 0.38;
    this.controls.maxDistance = 2.8;
    this.controls.maxPolarAngle = Math.PI / 2 + 0.04;
    this.reset();
    this.scene.add(new T.AmbientLight(0xffffff, 0.7));
    const hemi = new T.HemisphereLight(0xf3f8e9, 0x849272, 1.4);
    hemi.position.set(0, 0, 2);
    this.scene.add(hemi);
    const light = new T.DirectionalLight(0xffffff, 2.3);
    light.position.set(-0.6, -0.8, 1.8);
    light.castShadow = true;
    light.shadow.mapSize.set(1024, 1024);
    Object.assign(light.shadow.camera, {
      left: -1,
      right: 1,
      top: 1,
      bottom: -1,
      near: 0.1,
      far: 4,
    });
    light.shadow.bias = -0.0004;
    this.scene.add(light);
    const fill = new T.DirectionalLight(0xe5f1d2, 0.8);
    fill.position.set(1, 0.5, 0.7);
    this.scene.add(fill);
    this.bodies = data.poses.map(() => {
      const group = new T.Group();
      this.scene.add(group);
      return group;
    });
    for (const item of data.geometries) {
      let geometry;
      const [x, y, z] = item.size;
      if (item.type === 7) {
        if (!meshCache.has(item.mesh)) {
          const m = data.meshes[item.mesh],
            g = new T.BufferGeometry();
          g.setAttribute(
            "position",
            new T.BufferAttribute(unpack(m.vertices_b64, Float32Array), 3),
          );
          g.setIndex(
            new T.BufferAttribute(unpack(m.faces_b64, Uint32Array), 1),
          );
          g.computeVertexNormals();
          meshCache.set(item.mesh, g);
        }
        geometry = meshCache.get(item.mesh);
      } else if (item.type === 6)
        geometry = new T.BoxGeometry(2 * x, 2 * y, 2 * z);
      else if (item.type === 5) {
        geometry = new T.CylinderGeometry(x, x, 2 * y, 36);
        geometry.rotateX(Math.PI / 2);
      } else if (item.type === 2) geometry = new T.SphereGeometry(x, 24, 16);
      else if (item.type === 3) {
        geometry = new T.CapsuleGeometry(x, 2 * y, 5, 16);
        geometry.rotateX(Math.PI / 2);
      } else if (item.type === 4)
        geometry = new T.SphereGeometry(1, 24, 16).scale(x, y, z);
      else if (item.type === 0) geometry = new T.PlaneGeometry(4, 4);
      else continue;
      const color = new T.Color().setRGB(
        ...item.rgba.slice(0, 3),
        T.SRGBColorSpace,
      );
      const material = new T.MeshStandardMaterial({
        color,
        roughness: 0.72,
        metalness: 0.06,
        side: T.DoubleSide,
        transparent: item.rgba[3] < 1,
        opacity: item.rgba[3],
      });
      const mesh = new T.Mesh(geometry, material);
      mesh.position.fromArray(item.position);
      mesh.quaternion.copy(quaternion(item.quaternion));
      mesh.castShadow = item.type !== 0;
      mesh.receiveShadow = true;
      this.bodies[item.body].add(mesh);
    }
    this.setPoses(data.poses);
    const resize = () => {
      const { width, height } = element.getBoundingClientRect();
      if (!width || !height) return;
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height, false);
      this.dirty = true;
    };
    new ResizeObserver(resize).observe(element);
    resize();
    this.visible = true;
    new IntersectionObserver(([entry]) => {
      this.visible = entry.isIntersecting;
    }).observe(element);
    views.push(this);
  }
  reset(preset = "perspective") {
    this.controls.target.set(0, -0.15, 0.13);
    this.camera.position.set(
      ...{
        perspective: [0.93, -1.13, 0.85],
        top: [0.001, -0.15, 1.6],
        front: [0, -1.45, 0.58],
      }[preset],
    );
    this.controls.update();
  }
  setPoses(poses) {
    if (poses === this.previousPoses) return;
    this.previousPoses = poses;
    this.dirty = true;
    poses.forEach((p, i) => {
      const b = this.bodies[i];
      if (b) {
        b.position.set(...p.slice(0, 3));
        b.quaternion.copy(quaternion(p.slice(3)));
      }
    });
  }
  draw() {
    if (this.visible && !document.hidden) {
      this.controls.update();
      if (this.dirty) {
        this.renderer.render(this.scene, this.camera);
        this.dirty = false;
      }
    }
  }
}
export function drawViews(){views.forEach(v=>v.draw())}
let scenePromise;
export function loadScene(){return scenePromise??=fetch(new URL("./assets/scene.json.gz",import.meta.url)).then(async r=>{if(!r.ok)throw new Error("Scene unavailable");const bytes=new Uint8Array(await r.arrayBuffer());if(bytes[0]===31&&bytes[1]===139){const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));return new Response(stream).json()}return JSON.parse(new TextDecoder().decode(bytes))})}
