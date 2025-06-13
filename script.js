import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
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
let handState = {
    handDetected: false,
    isPinching: false,
    isPinkyExtended: false,
    explosionTriggered: false,

    // For locking the control reference frame
    initialPinch: true,
    initialHandPos: new THREE.Vector3(),
    initialAttractorPos: new THREE.Vector3(),
    cameraMatrixInverse: new THREE.Matrix4(),
    initialCameraQuaternion: new THREE.Quaternion()
};
let lastVideoTime = -1;
let debugCanvas, debugCtx, drawingUtils;
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
    if (!video.videoWidth) {
      requestAnimationFrame(predictWebcam);
      return;
    }

    if (handLandmarker) {
        const startTimeMs = performance.now();
        if (lastVideoTime !== video.currentTime) {
            lastVideoTime = video.currentTime;
            const results = handLandmarker.detectForVideo(video, startTimeMs);
            processHandLandmarks(results);
        }
    }
   
    window.requestAnimationFrame(predictWebcam);
}


function processHandLandmarks(results) {
    debugCtx.save();
    debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
    if (results.landmarks) {
        for (const landmarks of results.landmarks) {
            drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
            drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 2 });
        }
    }
    debugCtx.restore();

    handState.handDetected = results.landmarks && results.landmarks.length > 0;
    attractor.visible = handState.handDetected;

    if (!handState.handDetected) {
        if (handState.isPinching || handState.isPinkyExtended) {
            console.log("DEBUG: Hand lost. Resetting gestures.");
        }
        handState.isPinching = false;
        handState.isPinkyExtended = false;
        handState.initialPinch = true;
        return;
    }

    const landmarks = results.landmarks[0];
    const thumbTip = landmarks[4];
    const handX = (1 - thumbTip.x) * 2 - 1; 
    const handY = -(thumbTip.y * 2) + 1;

    const wrist = landmarks[0];
    const middleFingerMCP = landmarks[9];
    const handSize = Math.sqrt(Math.pow(wrist.x - middleFingerMCP.x, 2) + Math.pow(wrist.y - middleFingerMCP.y, 2));
    const minHandSize = 0.1, maxHandSize = 0.35, minDistance = 150, maxDistance = 400;
    const distRatio = THREE.MathUtils.inverseLerp(minHandSize, maxHandSize, handSize);
    const distanceFromCamera = THREE.MathUtils.lerp(minDistance, maxDistance, THREE.MathUtils.clamp(distRatio, 0, 1));
    
    raycaster.setFromCamera({ x: handX, y: handY }, mainCamera);
    const planeNormal = mainCamera.getWorldDirection(new THREE.Vector3());
    const planeOrigin = mainCamera.position.clone().add(planeNormal.clone().multiplyScalar(distanceFromCamera));
    interactionPlane.setFromNormalAndCoplanarPoint(planeNormal, planeOrigin);
    const currentHandPos = new THREE.Vector3();
    raycaster.ray.intersectPlane(interactionPlane, currentHandPos);

    if (handState.initialPinch) {
        console.log("DEBUG: New hand control session. Setting reference frame.");
        handState.initialPinch = false;
        attractor.position.copy(currentHandPos);
        handState.initialHandPos.copy(currentHandPos);
        handState.initialAttractorPos.copy(attractor.position);
        handState.cameraMatrixInverse.copy(mainCamera.matrixWorld).invert();
        handState.initialCameraQuaternion.copy(mainCamera.quaternion);
    } else {
        const currentHandLocal = currentHandPos.clone().applyMatrix4(handState.cameraMatrixInverse);
        const initialHandLocal = handState.initialHandPos.clone().applyMatrix4(handState.cameraMatrixInverse);
        const deltaLocal = currentHandLocal.sub(initialHandLocal);
        const deltaWorld = deltaLocal.clone().applyQuaternion(handState.initialCameraQuaternion);
        const newPos = handState.initialAttractorPos.clone().add(deltaWorld);
        attractor.position.lerp(newPos, 0.5);
    }
    
    // --- GESTURE ANALYSIS ---
    // 1. Pinch Gesture (for attraction)
    const indexTip = landmarks[8];
    const pinchDistance = Math.sqrt(Math.pow(thumbTip.x - indexTip.x, 2) + Math.pow(thumbTip.y - indexTip.y, 2));
    handState.isPinching = pinchDistance < 0.05;

    // 2. Extended Pinky Gesture (for explosion)
    const pinkyTip = landmarks[20];
    const pinkyBase = landmarks[17];
    const pinkyRingDist = Math.sqrt(Math.pow(pinkyTip.x - pinkyBase.x, 2) + Math.pow(pinkyTip.y - pinkyBase.y, 2));
    
    const wasExtended = handState.isPinkyExtended;
    handState.isPinkyExtended = pinkyRingDist > 0.08; 

    if (handState.isPinkyExtended && !wasExtended) {
        console.log("DEBUG: Extended pinky detected. Triggering explosion!");
        handState.explosionTriggered = true;
    }
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

        // [OPTIMIZATION] Pre-allocate vectors for calculations to avoid garbage collection
        this._cohesionVec = new THREE.Vector3();
        this._separationVec = new THREE.Vector3();
        this._alignmentVec = new THREE.Vector3();
        this._steerVec = new THREE.Vector3();
        this._tempVec = new THREE.Vector3();
    }

    applyForce(force) {
        this.acceleration.add(force);
    }
    
    // [OPTIMIZATION] Rewritten flock method using object pooling to reduce memory allocation
    flock(boidsInVicinity) {
        // Reset reusable vectors
        this._cohesionVec.set(0, 0, 0);
        this._separationVec.set(0, 0, 0);
        this._alignmentVec.set(0, 0, 0);

        let separationCount = 0;
        let alignmentCount = 0;
        let cohesionCount = 0;

        for (const other of boidsInVicinity) {
            if (other === this) continue;
            const d = this.position.distanceTo(other.position);

            if (d > 0 && d < boidParams.visualRange) {
                this._cohesionVec.add(other.position);
                cohesionCount++;
                this._alignmentVec.add(other.velocity);
                alignmentCount++;
            }

            if (d > 0 && d < boidParams.protectedRange) {
                this._tempVec.subVectors(this.position, other.position);
                this._tempVec.divideScalar(d * d);
                this._separationVec.add(this._tempVec);
                separationCount++;
            }
        }
        
        if (cohesionCount > 0) {
            this._cohesionVec.divideScalar(cohesionCount);
            this._tempVec.subVectors(this._cohesionVec, this.position);
            this._tempVec.setLength(boidParams.maxSpeed);
            this._steerVec.subVectors(this._tempVec, this.velocity);
            this._steerVec.clampLength(0, boidParams.maxForce);
            this.applyForce(this._steerVec.multiplyScalar(boidParams.cohesionForce));
        }

        if (alignmentCount > 0) {
            this._alignmentVec.divideScalar(alignmentCount);
            this._alignmentVec.setLength(boidParams.maxSpeed);
            this._steerVec.subVectors(this._alignmentVec, this.velocity);
            this._steerVec.clampLength(0, boidParams.maxForce);
            this.applyForce(this._steerVec.multiplyScalar(boidParams.alignmentForce));
        }

        if (separationCount > 0) {
            this._separationVec.divideScalar(separationCount);
            this._separationVec.setLength(boidParams.maxSpeed);
            this._steerVec.subVectors(this._separationVec, this.velocity);
            this._steerVec.clampLength(0, boidParams.maxForce);
            this.applyForce(this._steerVec.multiplyScalar(boidParams.separationForce));
        }

        if (handState.isPinching) {
            const attractorPos = attractor.position;
            this._tempVec.subVectors(attractorPos, this.position);
            const d = this._tempVec.length();
            if (d < boidParams.visualRange * 2) {
                this._tempVec.setLength(boidParams.maxSpeed);
                this._steerVec.subVectors(this._tempVec, this.velocity);
                this._steerVec.clampLength(0, boidParams.maxForce);
                this.applyForce(this._steerVec.multiplyScalar(boidParams.attractorForce));
            }
        }

        if (handState.explosionTriggered) {
            const explosionPos = attractor.position;
            const dist = this.position.distanceTo(explosionPos);
            const explosionRadius = 150;

            if (dist < explosionRadius && dist > 0) {
                this._tempVec.subVectors(this.position, explosionPos);
                const falloff = 1 - (dist / explosionRadius);
                const repulsionStrength = boidParams.maxForce * 1000 * falloff; 
                this._tempVec.setLength(repulsionStrength);
                this.applyForce(this._tempVec);
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
    // [OPTIMIZATION] Cap the pixel ratio for better performance on high-DPI screens
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
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

    const attractorGeometry = new THREE.SphereGeometry(8, 32, 32);
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
        
        if (handState.explosionTriggered) {
            handState.explosionTriggered = false;
        }

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
