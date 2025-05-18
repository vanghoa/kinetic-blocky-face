import p5 from 'p5';
import '@tensorflow/tfjs-backend-webgl';
import * as faceapi from '@vladmandic/face-api';
import * as bodySegmentation from '@tensorflow-models/body-segmentation';

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

function randomSaturatedColor() {
    const h = Math.floor(Math.random() * 360);
    const s = 70 + Math.floor(Math.random() * 31);
    const l = 45 + Math.floor(Math.random() * 16);
    function hslToRgb(h, s, l) {
        s /= 100;
        l /= 100;
        const c = (1 - Math.abs(2 * l - 1)) * s;
        const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
        const m = l - c / 2;
        let [r1, g1, b1] = [0, 0, 0];
        if (h < 60) [r1, g1, b1] = [c, x, 0];
        else if (h < 120) [r1, g1, b1] = [x, c, 0];
        else if (h < 180) [r1, g1, b1] = [0, c, x];
        else if (h < 240) [r1, g1, b1] = [0, x, c];
        else if (h < 300) [r1, g1, b1] = [x, 0, c];
        else [r1, g1, b1] = [c, 0, x];
        return [
            Math.round((r1 + m) * 255),
            Math.round((g1 + m) * 255),
            Math.round((b1 + m) * 255),
        ];
    }
    return hslToRgb(h, s, l);
}
const MOUTH_FILL = randomSaturatedColor();
const EYE_FILL = randomSaturatedColor();
const NOSE_FILL = randomSaturatedColor();
const OUTLINE_COLOR = randomSaturatedColor();

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
        let w = 700,
            h = 700,
            ratio = vw / vh;
        if (w / h > ratio) w = Math.round(h * ratio);
        else h = Math.round(w / ratio);
        p.createCanvas(w, h, p.WEBGL).parent('p5-container');
        video = p.createCapture(
            { video: { width: w, height: h }, audio: false },
            () => (videoReady = true)
        );
        video.hide();
        pg = p.createGraphics(w, h);
        pg.pixelDensity(1);
        // Start segmentation and landmark loops
        segmentationLoop();
        landmarkLoop(detectorOpts);
    };

    const THROTTLE = 200; // ms

    p.draw = (() => {
        let lastDraw = 0;
        return () => {
            const now = Date.now();
            if (
                !videoReady ||
                !segmenter ||
                !lastMaskData ||
                !lastMaskShape ||
                !lastBBox ||
                now - lastDraw < THROTTLE
            )
                return;
            lastDraw = now;
            // Only update hulls if new detections or mask shape
            if (landmarkUpdateNeeded || maskUpdateNeeded) {
                updateHulls(detections, lastMaskShape);
                landmarkUpdateNeeded = false;
                maskUpdateNeeded = false;
            }
            pg.image(video, 0, 0, pg.width, pg.height);
            pg.loadPixels();
            const buf = pg.pixels,
                bufW = pg.width;
            const { w, h, c } = lastMaskShape;
            const { minX: X0, minY: Y0, cropW, cropH } = lastBBox;
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
                        let r = buf[vidIdx],
                            g = buf[vidIdx + 1],
                            b = buf[vidIdx + 2];
                        if (rEyesHull && pointInPolygon(xVid, yVid, rEyesHull))
                            [r, g, b] = EYE_FILL;
                        else if (
                            lEyesHull &&
                            pointInPolygon(xVid, yVid, lEyesHull)
                        )
                            [r, g, b] = EYE_FILL;
                        else if (
                            noseHull &&
                            pointInPolygon(xVid, yVid, noseHull)
                        )
                            [r, g, b] = NOSE_FILL;
                        else if (
                            mouthHull &&
                            pointInPolygon(xVid, yVid, mouthHull)
                        )
                            [r, g, b] = MOUTH_FILL;
                        p.push();
                        p.fill(r, g, b);
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
