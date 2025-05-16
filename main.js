import p5 from 'p5';
import * as faceapi from 'face-api.js';

// block size ratio 10:20
const RECT_W = 40;
const RECT_H = 20;

// outline color (same for all features)

// distinct fills per feature
function randomSaturatedColor() {
    // H: 0-360, S: 60-100%, L: 40-60%
    const h = Math.floor(Math.random() * 360);
    const s = 70 + Math.floor(Math.random() * 31); // 70-100%
    const l = 45 + Math.floor(Math.random() * 16); // 45-60%
    return `hsl(${h},${s}%,${l}%)`;
}

const MOUTH_FILL = randomSaturatedColor();
const EYE_FILL = randomSaturatedColor();
const NOSE_FILL = randomSaturatedColor();
const OUTLINE_COLOR = randomSaturatedColor();

// how much of the canvas the face box should fill (0–1)
const BOX_FILL = 1;

// load models from /models
async function loadModels() {
    //await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
    await faceapi.loadTinyFaceDetectorModel('/kinetic-blocky-face/models');
    await faceapi.loadFaceLandmarkTinyModel('/kinetic-blocky-face/models');
    console.log('Models loaded');
}

// quantize points into grid cells
function quantizePoints(pts) {
    const cells = new Set();
    pts.forEach(({ x, y }) => {
        const col = Math.floor(x / RECT_W);
        const row = Math.floor(y / RECT_H);
        cells.add(`${col},${row}`);
    });
    return cells;
}

// draw a set of grid cells with a given fill
// Utility: Convex hull using Graham scan
function convexHull(points) {
    if (points.length < 3) return points.slice();
    points = points.slice().sort(([ax, ay], [bx, by]) => ax - bx || ay - by);
    const cross = (o, a, b) =>
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [];
    for (const p of points) {
        while (
            lower.length >= 2 &&
            cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0
        )
            lower.pop();
        lower.push(p);
    }
    const upper = [];
    for (let i = points.length - 1; i >= 0; i--) {
        const p = points[i];
        while (
            upper.length >= 2 &&
            cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0
        )
            upper.pop();
        upper.push(p);
    }
    upper.pop();
    lower.pop();
    return lower.concat(upper);
}

// Utility: Point-in-polygon (ray casting)
function pointInPolygon(x, y, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const [xi, yi] = polygon[i],
            [xj, yj] = polygon[j];
        if (
            yi > y !== yj > y &&
            x < ((xj - xi) * (y - yi)) / (yj - yi || 1e-10) + xi
        ) {
            inside = !inside;
        }
    }
    return inside;
}

// Draw a filled convex hull of grid cells
function drawCells(p, cells, fillColor) {
    if (!cells.size) return;
    p.stroke(OUTLINE_COLOR);
    p.strokeWeight(5);
    p.fill(fillColor);

    // Convert cell keys to [col, row]
    const cellCoords = Array.from(cells, (key) => key.split(',').map(Number));
    const hull = convexHull(cellCoords);

    // For fast lookup
    const cellSet = new Set(Array.from(cells));

    // Helper: is cell on hull outline?
    function isOnOutline(col, row) {
        // If cell is not inside hull, skip
        if (!pointInPolygon(col, row, hull)) return false;
        // If any 4-neighbor is outside hull, it's on outline
        const neighbors = [
            [col + 1, row],
            [col - 1, row],
            [col, row + 1],
            [col, row - 1],
        ];
        for (const [nc, nr] of neighbors) {
            if (!pointInPolygon(nc, nr, hull)) return true;
        }
        return false;
    }

    // Bounding box
    let minCol = Infinity,
        maxCol = -Infinity,
        minRow = Infinity,
        maxRow = -Infinity;
    for (const [col, row] of cellCoords) {
        if (col < minCol) minCol = col;
        if (col > maxCol) maxCol = col;
        if (row < minRow) minRow = row;
        if (row > maxRow) maxRow = row;
    }

    // Only fill cells on the outline of the hull
    for (let col = minCol; col <= maxCol; col++) {
        for (let row = minRow; row <= maxRow; row++) {
            if (isOnOutline(col, row)) {
                const x = col * RECT_W - p.width / 2 + RECT_W / 2;
                const y = row * RECT_H - p.height / 2 + RECT_H / 2;
                p.push();
                p.translate(x, y, 0);
                p.box(RECT_W, RECT_H, 100);
                p.pop();
            }
        }
    }
}

const sketch = (p) => {
    let video;
    let detections = null;
    const detectorOpts = new faceapi.TinyFaceDetectorOptions({
        inputSize: 160,
        scoreThreshold: 0.5,
    });

    p.setup = async () => {
        // Set canvas size to fit viewport ratio, but max 800x600
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        let w = 1000,
            h = 1000;
        const ratio = vw / vh;
        if (w / h > ratio) {
            w = Math.round(h * ratio);
        } else {
            h = Math.round(w / ratio);
        }
        // Use WEBGL mode for 3D
        p.createCanvas(w, h, p.WEBGL).parent('p5-container');
        video = p.createCapture(p.VIDEO);
        video.size(w, h);
        video.hide();
        await loadModels();
        requestAnimationFrame(detectLoop);
    };

    p.draw = () => {
        p.background('#fff8e7');
        if (!detections?.detection) return;

        // get face box and compute zoom
        const box = detections.detection.box;
        const cx = box.x + box.width / 2;
        const cy = box.y + box.height / 2;
        const scale = Math.max(
            (p.width * BOX_FILL) / box.width,
            (p.height * BOX_FILL) / box.height
        );

        // transform each landmark
        const transform = ({ x, y }) => ({
            x: (x - cx) * scale + p.width / 2,
            y: (y - cy) * scale + p.height / 2,
        });

        // quantize each feature
        const mouthCells = quantizePoints(
            detections.landmarks.getMouth().map(transform)
        );
        const leftEyeCells = quantizePoints(
            detections.landmarks.getLeftEye().map(transform)
        );
        const rightEyeCells = quantizePoints(
            detections.landmarks.getRightEye().map(transform)
        );
        const noseCells = quantizePoints(
            detections.landmarks.getNose().map(transform)
        );

        // Optionally, rotate scene for better 3D effect
        p.orbitControl();

        // draw each with its fill
        drawCells(p, mouthCells, MOUTH_FILL);
        drawCells(p, leftEyeCells, EYE_FILL);
        drawCells(p, rightEyeCells, EYE_FILL);
        drawCells(p, noseCells, NOSE_FILL);
    };

    // detection loop, throttled to ~10 FPS
    let lastTime = 0;
    const INTERVAL = 50;
    async function detectLoop(ts) {
        if (video.elt.readyState >= 2 && ts - lastTime > INTERVAL) {
            lastTime = ts;
            try {
                detections = await faceapi
                    .detectSingleFace(video.elt, detectorOpts)
                    .withFaceLandmarks(true);
            } catch (err) {
                console.error(err);
            }
        }
        requestAnimationFrame(detectLoop);
    }
};

document.getElementById('start-btn').addEventListener('click', async () => {
    const btn = document.getElementById('start-btn');
    btn.disabled = true;
    btn.textContent = 'LOADING...';

    // 1. Load TF backend if needed (face-api will auto-load tfjs)
    // 2. Load models
    await loadModels();

    // 3. Hide button, show canvas container
    btn.style.display = 'none';
    document.getElementById('p5-container').style.display = 'block';

    // 4. Start p5 sketch (getUserMedia now inside gesture)
    new p5(sketch);
});
