import p5 from 'p5';
import '@tensorflow/tfjs-backend-webgl';
import * as faceapi from '@vladmandic/face-api';
import * as bodySegmentation from '@tensorflow-models/body-segmentation';

// polyfilling
if (typeof window.OffscreenCanvas === 'undefined') {
    window.OffscreenCanvas = function (width, height) {
        const c = document.createElement('canvas');
        c.width = width;
        c.height = height;
        return c;
    };
}

const HEX_PALETTE = ['000000', 'FFFFFF', 'FCCBB4', 'F1A582', 'E0885F'];

const HEX_PALETTE_LANDMARK = [
    '2152FF',
    'FFE35B',
    'C351F4',
    '81D282',
    '5C9EFF',
    '9C0DDF',
    'CAAEFF',
    '9C5035',
];

// 2) helper to parse 3- or 6-digit hex into [r,g,b]
function hexToRgb(hex) {
    // strip leading “#”
    let h = hex.replace(/^#/, '');
    // expand short form “#0f8” → “00ff88”
    if (h.length === 3) {
        h = h
            .split('')
            .map((c) => c + c)
            .join('');
    }
    const intVal = parseInt(h, 16);
    return [(intVal >> 16) & 0xff, (intVal >> 8) & 0xff, intVal & 0xff];
}
function nearestColor(r, g, b) {
    let best = PALETTE[0];
    let bestDist = Infinity;
    for (const [pr, pg, pb] of PALETTE) {
        const dr = r - pr;
        const dg = g - pg;
        const db = b - pb;
        const d2 = dr * dr + dg * dg + db * db;
        if (d2 < bestDist) {
            bestDist = d2;
            best = [pr, pg, pb];
        }
    }
    return best;
}
// 3) build your numeric palette
const PALETTE = HEX_PALETTE.map(hexToRgb);
const PALETTE_LANDMARK = HEX_PALETTE_LANDMARK.map(hexToRgb);

const RECT_W = 20,
    RECT_H = 10,
    DO_FLIP = true;

let segmenter,
    detections = null,
    lastMaskData = null,
    lastMaskShape = null,
    lastBBox = null;
let lEyesHull, rEyesHull, noseHull, mouthHull;
let videoReady = false,
    video,
    pg;
let maskUpdateNeeded = false,
    landmarkUpdateNeeded = false;
let samplingCanvas, samplingCtx;

function getRandomPaletteColor(palette, used = new Set()) {
    let idx, color;
    do {
        idx = Math.floor(Math.random() * palette.length);
        color = palette[idx];
    } while (used.has(idx) && used.size < palette.length);
    used.add(idx);
    return color;
}
const usedColors = new Set();
const MOUTH_FILL = getRandomPaletteColor(PALETTE_LANDMARK, usedColors);
const EYE_FILL = getRandomPaletteColor(PALETTE_LANDMARK, usedColors);
const NOSE_FILL = getRandomPaletteColor(PALETTE_LANDMARK, usedColors);
const OUTLINE_COLOR = getRandomPaletteColor(PALETTE_LANDMARK, usedColors);
const OUTLINE_COLOR_LANDMARK = getRandomPaletteColor(
    PALETTE_LANDMARK,
    usedColors
);

async function loadModels() {
    await faceapi.loadTinyFaceDetectorModel('/kinetic-blocky-face/models');
    await faceapi.loadFaceLandmarkTinyModel('/kinetic-blocky-face/models');
    console.log('Models loaded');
}

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

function updateHulls(detections, maskShape) {
    if (!detections?.landmarks || !maskShape) return;
    const { w: mw, h: mh } = maskShape;
    const vidW = video.elt.videoWidth,
        vidH = video.elt.videoHeight;
    const scaleX = mw / vidW,
        scaleY = mh / vidH;
    function project(pt) {
        return [pt.x * scaleX, pt.y * scaleY];
    }
    rEyesHull = convexHull(detections.landmarks.getRightEye().map(project));
    lEyesHull = convexHull(detections.landmarks.getLeftEye().map(project));
    noseHull = convexHull(detections.landmarks.getNose().map(project));
    mouthHull = convexHull(detections.landmarks.getMouth().map(project));
}

// ─── p5 Sketch ────────────────────────────────────────────────────
const sketch = (p) => {
    const detectorOpts = new faceapi.TinyFaceDetectorOptions({
        inputSize: 160,
        scoreThreshold: 0.5,
    });

    p.setup = async () => {
        const vw = window.innerWidth,
            vh = window.innerHeight;
        let ogw = 700,
            ogh = 800;
        let w = ogw,
            h = ogh,
            ratio = vw / vh;
        if (w / h > ratio) w = Math.round(h * ratio);
        else h = Math.round(w / ratio);
        //console.log(w, h);
        p.pixelDensity(1);
        p.createCanvas(w, h, p.WEBGL).parent('p5-container');
        video = p.createCapture(
            { video: { width: ogw, height: ogh }, audio: false },
            () => (videoReady = true)
        );
        video.hide();

        samplingCanvas = document.createElement('canvas');
        samplingCanvas.width = ogw;
        samplingCanvas.height = ogh;
        samplingCtx = samplingCanvas.getContext('2d');
        pg = p.createGraphics(ogw, ogh);
        pg.pixelDensity(1);
        // Start segmentation and landmark loops
        segmentationLoop();
        landmarkLoop(detectorOpts);
    };

    const THROTTLE = 400; // ms
    let init = true;

    p.draw = (() => {
        let lastDraw = 0;
        let buf = null;
        let bufW = 20;
        return () => {
            const now = Date.now();
            const vw = video.elt.videoWidth;
            const vh = video.elt.videoHeight;
            if (now - lastDraw >= THROTTLE && vw) {
                // paint the current video frame into your 2D canvas
                if (
                    samplingCanvas.width !== vw ||
                    samplingCanvas.height !== vh
                ) {
                    samplingCanvas.width = vw;
                    samplingCanvas.height = vh;
                }
                samplingCtx.drawImage(video.elt, 0, 0, vw, vh);
                const imageData = samplingCtx.getImageData(0, 0, vw, vh);
                buf = imageData.data; // Uint8ClampedArray of RGBA
                bufW = vw;
                lastDraw = now;
            }
            if (
                !videoReady ||
                !segmenter ||
                !lastMaskData ||
                !lastMaskShape ||
                !lastBBox ||
                !buf ||
                !bufW
            )
                return;
            p.orbitControl();
            // Only update hulls if new detections or mask shape
            if (landmarkUpdateNeeded || maskUpdateNeeded) {
                updateHulls(detections, lastMaskShape);
                landmarkUpdateNeeded = false;
                maskUpdateNeeded = false;
            }
            //pg.image(video, 0, 0, pg.width, pg.height);
            const { w, h, c } = lastMaskShape;
            const { minX: X0, minY: Y0, cropW, cropH } = lastBBox;
            // Only zoom out at the start, then let orbitControl take over
            if (init) {
                init = false;
                // Randomly position camera on a sphere of radius 2000, horizontally from -1500 to 1500, vertically from -1000 to 1000, always looking at the center
                const isMobile = window.innerWidth < window.innerHeight;
                const radius = isMobile ? 1000 : 2000;
                // Limit rotation: camX and camY are proportional to radius, with lower rotation on mobile
                const limit = isMobile ? 0.4 : 0.75;
                const camX = (Math.random() * (2 * limit) - limit) * radius;
                const camY = (Math.random() * (2 * limit) - limit) * radius;
                // Only allow camZ to be positive (front side)
                const camZ = Math.sqrt(
                    Math.max(0, radius * radius - camX * camX - camY * camY)
                );
                p.camera(
                    camX,
                    camY,
                    camZ,
                    0, // look at center
                    0,
                    0,
                    0,
                    1,
                    0
                );
            }
            p.background(255);
            p.stroke(OUTLINE_COLOR);
            p.strokeWeight(1);
            for (let yy = 0; yy < cropH; yy += RECT_H) {
                for (let xx = 0; xx < cropW; xx += RECT_W) {
                    const rawX = X0 + xx + RECT_W / 2;
                    const xVid = DO_FLIP ? w - 1 - rawX : rawX;
                    const yVid = Y0 + yy + RECT_H / 2;
                    const maskIdx = (yVid * w + xVid) * c;
                    if (lastMaskData[maskIdx] > 0.5) {
                        const vidIdx = (yVid * bufW + xVid) * 4;
                        let isLandmark = false;
                        let r = buf[vidIdx],
                            g = buf[vidIdx + 1],
                            b = buf[vidIdx + 2];
                        if (
                            rEyesHull &&
                            pointInPolygon(xVid, yVid, rEyesHull)
                        ) {
                            isLandmark = true;
                            [r, g, b] = EYE_FILL;
                        } else if (
                            lEyesHull &&
                            pointInPolygon(xVid, yVid, lEyesHull)
                        ) {
                            isLandmark = true;
                            [r, g, b] = EYE_FILL;
                        } else if (
                            noseHull &&
                            pointInPolygon(xVid, yVid, noseHull)
                        ) {
                            isLandmark = true;
                            [r, g, b] = NOSE_FILL;
                        } else if (
                            mouthHull &&
                            pointInPolygon(xVid, yVid, mouthHull)
                        ) {
                            isLandmark = true;
                            [r, g, b] = MOUTH_FILL;
                        } else {
                            [r, g, b] = nearestColor(r, g, b);
                        }
                        p.push();
                        p.fill(r, g, b);
                        if (isLandmark) {
                            p.stroke(OUTLINE_COLOR_LANDMARK);
                            p.strokeWeight(2);
                        }
                        p.translate(
                            xx - cropW / 2 + RECT_W / 2,
                            yy - cropH / 2 + RECT_H / 2,
                            0
                        );
                        p.box(RECT_W, RECT_H, 100);
                        p.pop();
                    }
                }
            }
        };
    })();

    // Use the same THROTTLE for segmentation and landmark loops
    function segmentationLoop() {
        if (videoReady && segmenter) {
            segmenter.segmentPeople(video.elt).then(([seg]) => {
                let maskTensor = seg.mask.toTensor
                    ? seg.mask.toTensor()
                    : seg.mask;
                Promise.resolve(maskTensor).then(async (tensor) => {
                    const maskData = await tensor.data();
                    const [mh, mw, mc] = tensor.shape;
                    let minX = mw,
                        minY = mh,
                        maxX = 0,
                        maxY = 0;
                    for (let i = 0; i < maskData.length; i += mc) {
                        if (maskData[i] > 0.5) {
                            const idx = i / mc;
                            const y = Math.floor(idx / mw),
                                x = idx % mw;
                            minX = Math.min(minX, x);
                            minY = Math.min(minY, y);
                            maxX = Math.max(maxX, x);
                            maxY = Math.max(maxY, y);
                        }
                    }
                    lastMaskData = maskData;
                    lastMaskShape = { w: mw, h: mh, c: mc };
                    lastBBox = {
                        minX,
                        minY,
                        cropW: maxX - minX,
                        cropH: maxY - minY,
                    };
                    tensor.dispose && tensor.dispose();
                    maskUpdateNeeded = true;
                });
            });
        }
        setTimeout(segmentationLoop, THROTTLE);
    }

    function landmarkLoop(detectorOpts) {
        if (videoReady) {
            faceapi
                .detectSingleFace(video.elt, detectorOpts)
                .withFaceLandmarks(true)
                .then((res) => {
                    detections = res;
                    landmarkUpdateNeeded = true;
                })
                .catch(() => {});
        }
        setTimeout(() => landmarkLoop(detectorOpts), THROTTLE);
    }
};

document.getElementById('start-btn').addEventListener('click', async () => {
    const btn = document.getElementById('start-btn');
    btn.disabled = true;
    btn.textContent = 'LOADING...';
    await loadModels();
    segmenter = await bodySegmentation.createSegmenter(
        bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation,
        {
            runtime: 'mediapipe',
            solutionPath:
                'https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation',
            modelType: 'general',
        }
    );
    btn.style.display = 'none';
    document.getElementById('p5-container').style.display = 'block';
    new p5(sketch);
});
