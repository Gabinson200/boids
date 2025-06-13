import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
// [MODIFIED] Added DrawingUtils for visualization
import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://esm.sh/@mediapipe/tasks-vision@0.10.10";

// --- GLOBAL STATE ---
let scene, mainCamera, renderer, mainControls;
let boids = [];
let isPaused = false;
let hashGrid;

// Hand tracking variables
let handLandmarker;
let video;
let attractor;
let isPinching = false;
let lastVideoTime = -1;
let debugCanvas, debugCtx, drawingUtils; 
// [FIXED] Add raycaster and interaction plane for robust coordinate mapping
const raycaster = new THREE.Raycaster();
const interactionPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);


const bounds = new THREE.Vector3(200, 200, 200);
const boidParams = {
    count: 500,
    cohesionForce: 0.2,
    separationForce: 1.5,
    alignmentForce: 0.5,
    visualRange: 50,
    protectedRange: 10,
    maxSpeed: 2.0,
    minSpeed: 0.5,
    maxForce: 0.1,
    attractorForce: 0.8,
};

const dummy = new THREE.Object3D();
let instancedBoids;

// --- HAND TRACKING SETUP ---
async function setupHandTracking() {
    console.log("DEBUG: Starting hand tracking setup...");
    const loadingText = document.getElementById('loading-text');
    loadingText.textContent = 'Initializing Hand Tracking...';
    
    debugCanvas = document.getElementById("debug-canvas");
    debugCtx = debugCanvas.getContext("2d");
    drawingUtils = new DrawingUtils(debugCtx);


    const vision = await FilesetResolver.forVisionTasks(
        "https://unpkg.com/@mediapipe/tasks-vision@0.10.10/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });
    console.log("DEBUG: HandLandmarker created.");

    video = document.getElementById("webcam");
    const constraints = { video: { width: 640, height: 480 } };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        // [FIXED] Add an event listener to start prediction only when the video is playing
        video.addEventListener("loadeddata", predictWebcam);
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                debugCanvas.width = video.videoWidth;
                debugCanvas.height = video.videoHeight;
                console.log("DEBUG: Webcam loaded and ready.");
                resolve();
            };
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
        loadingText.textContent = 'Webcam access denied. Cannot use attractor.';
        setTimeout(() => {
            const overlay = document.getElementById('loading-overlay');
            if(overlay) overlay.style.display = 'none';
        }, 3000);
        return Promise.reject(error);
    }
}

function predictWebcam() {
    // [FIXED] Restructured the prediction loop for continuous processing
    if (!video.videoWidth) {
      requestAnimationFrame(predictWebcam);
      return;
    }

    if (handLandmarker) {
        const startTimeMs = performance.now();
        if (lastVideoTime !== video.currentTime) {
            lastVideoTime = video.currentTime;
            const results = handLandmarker.detectForVideo(video, startTimeMs);
             // Process the results
            processHandLandmarks(results);
        }
    }
   
    // Continue the loop
    window.requestAnimationFrame(predictWebcam);
}


function processHandLandmarks(results) {
    debugCtx.save();
    debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
    if (results.landmarks) {
        for (const landmarks of results.landmarks) {
            drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
                color: "#00FF00",
                lineWidth: 5
            });
            drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 2 });
        }
    }
    debugCtx.restore();
    
    if (results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];
        
        const thumbTip = landmarks[4];
        const indexTip = landmarks[8];
        
        const distance = Math.sqrt(
            Math.pow(thumbTip.x - indexTip.x, 2) +
            Math.pow(thumbTip.y - indexTip.y, 2) +
            Math.pow(thumbTip.z - indexTip.z, 2)
        );

        const pinchThreshold = 0.05;
        isPinching = distance < pinchThreshold;
        
        if (isPinching) {
            // --- [NEW] Z-depth control logic ---

            // 1. Calculate apparent hand size for depth perception.
            // We measure the 2D distance between the wrist and the middle finger knuckle.
            const wrist = landmarks[0];
            const middleFingerMCP = landmarks[9];
            const dx = wrist.x - middleFingerMCP.x;
            const dy = wrist.y - middleFingerMCP.y;
            const handSize = Math.sqrt(dx * dx + dy * dy);

            // 2. Map hand size to a distance from the camera.
            // These values can be tweaked for better feel and responsiveness.
            const minHandSize = 0.1;    // Represents hand being far away
            const maxHandSize = 0.35;   // Represents hand being very close
            const minDistance = 150;    // How far the attractor is at minHandSize
            const maxDistance = 400;    // How close the attractor is at maxHandSize

            // Map the size to a 0-1 ratio.
            const distRatio = THREE.MathUtils.inverseLerp(minHandSize, maxHandSize, handSize);
            // Lerp the distance, clamping the ratio to handle edge cases.
            const distanceFromCamera = THREE.MathUtils.lerp(minDistance, maxDistance, THREE.MathUtils.clamp(distRatio, 0, 1));


            // --- [MODIFIED] Use the new depth to position the interaction plane ---
            const handX = (1 - thumbTip.x) * 2 - 1; 
            const handY = -(thumbTip.y * 2) + 1;
            
            raycaster.setFromCamera({ x: handX, y: handY }, mainCamera);
            const pos = new THREE.Vector3();

            // Create an interaction plane at the new dynamic distance in front of the camera.
            const planeNormal = mainCamera.getWorldDirection(new THREE.Vector3());
            const planeOrigin = mainCamera.position.clone().add(planeNormal.clone().multiplyScalar(distanceFromCamera));
            interactionPlane.setFromNormalAndCoplanarPoint(planeNormal, planeOrigin);
            
            raycaster.ray.intersectPlane(interactionPlane, pos);
            
            attractor.position.lerp(pos, 0.5);

            // The existing color-changing logic will now work correctly with the new depth
            const distanceToCameraColor = attractor.position.distanceTo(mainCamera.position);
            const minColorDist = 100;
            const maxColorDist = 400;
            const depthRatioColor = THREE.MathUtils.clamp(
                THREE.MathUtils.inverseLerp(minColorDist, maxColorDist, distanceToCameraColor),
                0,
                1
            );
            const closeColor = new THREE.Color(0xff4400);
            const farColor = new THREE.Color(0x00aaff);
            attractor.material.color.lerpColors(closeColor, farColor, depthRatioColor);
        }

    } else {
        isPinching = false;
    }
    
    attractor.visible = isPinching;
}


// --- OPTIMIZATION: HASH GRID CLASS ---
class HashGrid {
    constructor(bounds, cellSize) {
        this.bounds = bounds;
        this.cellSize = cellSize;
        this.grid = new Map();
    }

    getGridKey(position) {
        const ix = Math.floor((position.x + this.bounds.x) / this.cellSize);
        const iy = Math.floor((position.y + this.bounds.y) / this.cellSize);
        const iz = Math.floor((position.z + this.bounds.z) / this.cellSize);
        return `${ix},${iy},${iz}`;
    }

    update(boids) {
        this.grid.clear();
        for (const boid of boids) {
            const key = this.getGridKey(boid.position);
            if (!this.grid.has(key)) {
                this.grid.set(key, []);
            }
            this.grid.get(key).push(boid);
        }
    }

    query(position, radius) {
        const neighbors = [];
        const pMin = new THREE.Vector3().copy(position).subScalar(radius);
        const pMax = new THREE.Vector3().copy(position).addScalar(radius);

        const iMinX = Math.floor((pMin.x + this.bounds.x) / this.cellSize);
        const iMaxX = Math.floor((pMax.x + this.bounds.x) / this.cellSize);
        const iMinY = Math.floor((pMin.y + this.bounds.y) / this.cellSize);
        const iMaxY = Math.floor((pMax.y + this.bounds.y) / this.cellSize);
        const iMinZ = Math.floor((pMin.z + this.bounds.z) / this.cellSize);
        const iMaxZ = Math.floor((pMax.z + this.bounds.z) / this.cellSize);

        for (let iz = iMinZ; iz <= iMaxZ; iz++) {
            for (let iy = iMinY; iy <= iMaxY; iy++) {
                for (let ix = iMinX; ix <= iMaxX; ix++) {
                    const key = `${ix},${iy},${iz}`;
                    if (this.grid.has(key)) {
                        neighbors.push(...this.grid.get(key));
                    }
                }
            }
        }
        return neighbors;
    }
}

// --- BOID CLASS ---
class Boid {
    constructor() {
        this.position = new THREE.Vector3(
            THREE.MathUtils.randFloatSpread(bounds.x * 2),
            THREE.MathUtils.randFloatSpread(bounds.y * 2),
            THREE.MathUtils.randFloatSpread(bounds.z * 2)
        );
        this.velocity = new THREE.Vector3(
            THREE.MathUtils.randFloatSpread(2),
            THREE.MathUtils.randFloatSpread(2),
            THREE.MathUtils.randFloatSpread(2)
        ).setLength(THREE.MathUtils.randFloat(boidParams.minSpeed, boidParams.maxSpeed));
        this.acceleration = new THREE.Vector3();
    }

    applyForce(force) {
        this.acceleration.add(force);
    }
    
    flock(boidsInVicinity) {
        const separation = new THREE.Vector3();
        const alignment = new THREE.Vector3();
        const cohesion = new THREE.Vector3();
        let separationCount = 0;
        let alignmentCount = 0;
        let cohesionCount = 0;

        for (const other of boidsInVicinity) {
            if (other === this) continue;
            const d = this.position.distanceTo(other.position);

            if (d > 0 && d < boidParams.visualRange) {
                cohesion.add(other.position);
                cohesionCount++;
                alignment.add(other.velocity);
                alignmentCount++;
            }

            if (d > 0 && d < boidParams.protectedRange) {
                const diff = new THREE.Vector3().subVectors(this.position, other.position);
                diff.divideScalar(d * d);
                separation.add(diff);
                separationCount++;
            }
        }
        
        if (cohesionCount > 0) {
            cohesion.divideScalar(cohesionCount);
            const desired = new THREE.Vector3().subVectors(cohesion, this.position);
            desired.setLength(boidParams.maxSpeed);
            const steer = new THREE.Vector3().subVectors(desired, this.velocity);
            steer.clampLength(0, boidParams.maxForce);
            this.applyForce(steer.multiplyScalar(boidParams.cohesionForce));
        }

        if (alignmentCount > 0) {
            alignment.divideScalar(alignmentCount);
            alignment.setLength(boidParams.maxSpeed);
            const steer = new THREE.Vector3().subVectors(alignment, this.velocity);
            steer.clampLength(0, boidParams.maxForce);
            this.applyForce(steer.multiplyScalar(boidParams.alignmentForce));
        }

        if (separationCount > 0) {
            separation.divideScalar(separationCount);
            separation.setLength(boidParams.maxSpeed);
            const steer = new THREE.Vector3().subVectors(separation, this.velocity);
            steer.clampLength(0, boidParams.maxForce);
            this.applyForce(steer.multiplyScalar(boidParams.separationForce));
        }

        if (isPinching) {
            const attractorPos = attractor.position;
            const desired = new THREE.Vector3().subVectors(attractorPos, this.position);
            const d = desired.length();
            if (d < boidParams.visualRange * 2) {
                desired.setLength(boidParams.maxSpeed);
                const steer = new THREE.Vector3().subVectors(desired, this.velocity);
                steer.clampLength(0, boidParams.maxForce);
                this.applyForce(steer.multiplyScalar(boidParams.attractorForce));
            }
        }
    }
    
    update() {
        this.velocity.add(this.acceleration);
        this.velocity.clampLength(boidParams.minSpeed, boidParams.maxSpeed);
        this.position.add(this.velocity);
        this.acceleration.multiplyScalar(0);
        this.wrapBounds();
    }
    
    wrapBounds() {
        if (this.position.x < -bounds.x) this.position.x = bounds.x;
        if (this.position.x > bounds.x) this.position.x = -bounds.x;
        if (this.position.y < -bounds.y) this.position.y = bounds.y;
        if (this.position.y > bounds.y) this.position.y = -bounds.y;
        if (this.position.z < -bounds.z) this.position.z = bounds.z;
        if (this.position.z > bounds.z) this.position.z = -bounds.z;
    }
}

// --- INITIALIZATION ---
async function init() {
    
    const canvasContainer = document.getElementById('canvas-container');
    
    scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x11111a, 100, 2000);

    mainCamera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 3000);
    mainCamera.position.set(0, 50, 300);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x000000, 0);
    canvasContainer.appendChild(renderer.domElement);
    
    const background = document.createElement('div');
    background.style.position = 'absolute';
    background.style.top = '0';
    background.style.left = '0';
    background.style.width = '100%';
    background.style.height = '100%';
    background.style.background = '#111827';
    background.style.zIndex = '-1';
    canvasContainer.appendChild(background);


    mainControls = new OrbitControls(mainCamera, renderer.domElement);
    mainControls.enableDamping = true;
    mainControls.dampingFactor = 0.05;
    
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    initUI();

    // [MODIFIED] Increased geometry segments for a smoother sphere
    const attractorGeometry = new THREE.SphereGeometry(8, 32, 32);
    // [MODIFIED] Switched to MeshStandardMaterial for realistic 3D shading
    const attractorMaterial = new THREE.MeshStandardMaterial({
        color: 0x00ffaa,
        roughness: 0.2,
        metalness: 0.9,
        transparent: true,
        opacity: 0.8
    });
    attractor = new THREE.Mesh(attractorGeometry, attractorMaterial);
    attractor.visible = false;
    scene.add(attractor);

    const boidGeometry = new THREE.ConeGeometry(2, 8, 4); 
    boidGeometry.rotateX(Math.PI / 2);
    const boidMaterial = new THREE.MeshLambertMaterial({ color: 0xeeeeff });
    instancedBoids = new THREE.InstancedMesh(boidGeometry, boidMaterial, boidParams.count);
    scene.add(instancedBoids);

    const boundsBox = new THREE.Box3().setFromCenterAndSize(new THREE.Vector3(0,0,0), bounds.clone().multiplyScalar(2));
    const boundsHelper = new THREE.Box3Helper(boundsBox, 0x444444);
    scene.add(boundsHelper);
    
    hashGrid = new HashGrid(bounds, boidParams.visualRange);
    for (let i = 0; i < boidParams.count; i++) {
        boids.push(new Boid());
    }

    try {
        setupHandTracking();
        console.log("DEBUG: Hand tracking setup initiated.");
    } catch(e) {
        console.log("DEBUG: Could not initialize hand tracking.")
    }

    document.getElementById('loading-overlay').style.display = 'none';
    animate();
}

// --- UI INITIALIZATION ---
function initUI() {
    const sliders = {
        'attractor-force': 'attractorForce',
        'cohesion': 'cohesionForce',
        'separation': 'separationForce',
        'alignment': 'alignmentForce',
        'visual-range': 'visualRange',
        'protected-range': 'protectedRange'
    };
    
    const urlParams = new URLSearchParams(window.location.search);
    boidParams.count = parseInt(urlParams.get('count') || '500');
    const boidCountSlider = document.getElementById('boid-count');
    const boidCountValue = document.getElementById('boid-count-value');
    boidCountSlider.value = boidParams.count;
    boidCountValue.textContent = boidParams.count;

    boidCountSlider.addEventListener('change', (e) => {
        const newCount = e.target.value;
        const currentUrl = new URL(window.location.href);
        currentUrl.searchParams.set('count', newCount);
        window.location.href = currentUrl.href;
    });
    boidCountSlider.addEventListener('input', (e) => {
        boidCountValue.textContent = e.target.value;
    });

    for (const [id, param] of Object.entries(sliders)) {
        const slider = document.getElementById(id);
        const valueSpan = document.getElementById(`${id}-value`);
        
        slider.value = boidParams[param];
        if (valueSpan) valueSpan.textContent = boidParams[param].toFixed(2);

        slider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            boidParams[param] = value;
            if (valueSpan) valueSpan.textContent = value.toFixed(2);
            if (param === 'visualRange') {
                hashGrid = new HashGrid(bounds, value);
            }
        });
    }

    const pauseButton = document.getElementById('pause-button');
    pauseButton.addEventListener('click', () => {
        isPaused = !isPaused;
        pauseButton.textContent = isPaused ? 'Resume' : 'Pause';
    });
    
    const controlPanel = document.getElementById('control-panel');
    const collapseButton = document.getElementById('collapse-button');
    let isCollapsed = false;
    collapseButton.addEventListener('click', () => {
        isCollapsed = !isCollapsed;
        controlPanel.classList.toggle('collapsed', isCollapsed);
        collapseButton.style.left = isCollapsed ? '0px' : '320px';
        collapseButton.innerHTML = isCollapsed ? '&gt;' : '&lt;';
    });
}

// --- ANIMATION LOOP ---
function animate() {
    requestAnimationFrame(animate);

    if (!isPaused) {
        hashGrid.update(boids);
        
        boids.forEach(boid => {
            const neighbors = hashGrid.query(boid.position, boidParams.visualRange);
            boid.flock(neighbors);
        });
        
        boids.forEach((boid, i) => {
            boid.update();
            dummy.position.copy(boid.position);
            dummy.lookAt(boid.position.clone().add(boid.velocity));
            dummy.updateMatrix();
            instancedBoids.setMatrixAt(i, dummy.matrix);
        });

        instancedBoids.instanceMatrix.needsUpdate = true;
    }

    mainControls.update();
    render();
}

// --- RENDER FUNCTION ---
function render() {
    renderer.render(scene, mainCamera);
}

// --- EVENT LISTENERS ---
window.addEventListener('resize', () => {
    mainCamera.aspect = window.innerWidth / window.innerHeight;
    mainCamera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- START ---
window.addEventListener('DOMContentLoaded', () => {
    init();
});
