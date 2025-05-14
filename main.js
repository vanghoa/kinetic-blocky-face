import p5 from 'p5';
import * as faceapi from 'face-api.js';

// block size ratio 10:20
const RECT_W = 40;
const RECT_H = 20;

// outline color (same for all features)
const OUTLINE_COLOR = '#8B4513';

// distinct fills per feature
const MOUTH_FILL = '#d2691e'; // chocolate
const EYE_FILL = '#6495ED'; // steelblue
const NOSE_FILL = '#90EE90'; // lightgreen

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
function drawCells(p, cells, fillColor) {
    p.stroke(OUTLINE_COLOR);
    p.strokeWeight(3);
    p.fill(fillColor);

    // 3D effect: draw extruded blocks using p5's WEBGL mode
    // Assume p is in WEBGL mode (canvas created with WEBGL)
    cells.forEach((key) => {
        const [col, row] = key.split(',').map(Number);
        const x = col * RECT_W - p.width / 2 + RECT_W / 2;
        const y = row * RECT_H - p.height / 2 + RECT_H / 2;
        const z = 0;
        p.push();
        p.translate(x, y, z);
        p.box(RECT_W, RECT_H, 100); // 20 is the extrusion depth
        p.pop();
    });
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
    const INTERVAL = 100;
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
    btn.textContent = 'Loading…';

    // 1. Load TF backend if needed (face-api will auto-load tfjs)
    // 2. Load models
    await loadModels();

    // 3. Hide button, show canvas container
    btn.style.display = 'none';
    document.getElementById('p5-container').style.display = 'block';

    // 4. Start p5 sketch (getUserMedia now inside gesture)
    new p5(sketch);
});
