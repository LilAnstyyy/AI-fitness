import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

/* =========================
   DOM
========================= */
const video = document.getElementById("webcam");
const canvas = document.getElementById("output_canvas");
const ctx = canvas.getContext("2d");

const repCountEl = document.getElementById("repCount");
const timerEl = document.getElementById("timer");
const feedbackEl = document.getElementById("feedback");
const exerciseNameEl = document.getElementById("exerciseName");

const photoPreview = document.getElementById("photoPreview");
const photoFeedbackEl = document.getElementById("photoFeedback");
const photoExerciseNameEl = document.getElementById("photoExerciseName");

const startBtn = document.getElementById("startButton");
const stopBtn = document.getElementById("stopButton");
const resetBtn = document.getElementById("resetButton");

/* =========================
   APP STATE
========================= */
const AppState = {
  mode: "idle",            // idle | video | photo
  exercise: "auto",
  cameraRunning: false,
  drawingEnabled: true
};

/* =========================
   MODEL & STREAM
========================= */
let poseLandmarker = null;
let stream = null;

/* =========================
   COUNTERS
========================= */
let repCount = 0;
let plankStart = 0;
let lastRepTime = 0;

let squatStage = "up";
let lungeStage = "up";
let pushupStage = "up";

const MIN_REP_INTERVAL = 800;

/* =========================
   CONSTANTS
========================= */
const EXERCISE_NAMES = {
  auto: "Автоопределение",
  squats: "Приседания",
  lunges: "Выпады",
  plank: "Планка",
  pushups: "Отжимания",
  none: "Стойка"
};

/* =========================
   INIT
========================= */
document.addEventListener("DOMContentLoaded", async () => {
  canvas.width = 640;
  canvas.height = 480;
  renderIdle();
  initToggleSkeletonButton();
  updateControls();
});

/* =========================
   MODEL LOADING
========================= */
async function loadModel() {
  if (poseLandmarker) return;

  feedbackEl.textContent = "Загрузка модели ИИ…";
  photoFeedbackEl.textContent = "Загрузка модели ИИ…";

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numPoses: 1
  });

  feedbackEl.textContent = "Модель загружена. Включите камеру или загрузите фото.";
  photoFeedbackEl.textContent = "Модель загружена. Загрузите фото.";
}

/* =========================
   UI HELPERS
========================= */
function updateControls() {
  startBtn.disabled = AppState.cameraRunning;
  stopBtn.disabled = !AppState.cameraRunning;

  const toggle = document.getElementById("toggleSkeleton");
  if (toggle) toggle.disabled = !AppState.cameraRunning;
}

function resetCounters() {
  repCount = 0;
  plankStart = 0;
  lastRepTime = 0;

  squatStage = "up";
  lungeStage = "up";
  pushupStage = "up";

  repCountEl.textContent = "0";
  timerEl.textContent = "0";
}

function renderIdle() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#333";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#fff";
  ctx.font = "20px Arial";
  ctx.textAlign = "center";
  ctx.fillText("Включите камеру или загрузите фото", canvas.width / 2, canvas.height / 2);
}

/* =========================
   TOGGLE SKELETON
========================= */
function initToggleSkeletonButton() {
  if (document.getElementById("toggleSkeleton")) return;

  const controls = document.querySelector(".controls");
  if (!controls) return;

  const btn = document.createElement("button");
  btn.id = "toggleSkeleton";
  btn.className = "btn btn-secondary";
  btn.textContent = "Скрыть скелет";
  btn.disabled = true;

  btn.onclick = () => {
    AppState.drawingEnabled = !AppState.drawingEnabled;
    btn.textContent = AppState.drawingEnabled ? "Скрыть скелет" : "Показать скелет";
  };

  controls.appendChild(btn);
}

/* =========================
   GEOMETRY
========================= */
function angle(a, b, c) {
  const r =
    Math.atan2(c.y - b.y, c.x - b.x) -
    Math.atan2(a.y - b.y, a.x - b.x);
  let deg = Math.abs((r * 180) / Math.PI);
  return deg > 180 ? 360 - deg : deg;
}

function bodyHorizontal(lm) {
  return Math.abs(
    (lm[11].y + lm[12].y) / 2 -
    (lm[23].y + lm[24].y) / 2
  ) < 0.1;
}

/* =========================
   EXERCISE DETECTION
========================= */
function detectExercise(lm) {
  const lK = angle(lm[23], lm[25], lm[27]);
  const rK = angle(lm[24], lm[26], lm[28]);
  const avgK = (lK + rK) / 2;

  const lE = angle(lm[11], lm[13], lm[15]);
  const rE = angle(lm[12], lm[14], lm[16]);
  const avgE = (lE + rE) / 2;

  if (avgE < 120 && bodyHorizontal(lm)) return "pushups";
  if (avgK < 135 && Math.abs(lK - rK) > 30) return "lunges";
  if (avgK < 135) return "squats";
  if (avgK > 160 && bodyHorizontal(lm)) return "plank";

  return "none";
}

/* =========================
   FEEDBACK & COUNTING
========================= */
function handleExercise(ex, lm) {
  let msg = "";
  let color = "#ffcc00";

  if (ex === "squats") {
    const k = (angle(lm[23], lm[25], lm[27]) + angle(lm[24], lm[26], lm[28])) / 2;
    if (k < 95 && squatStage === "up") squatStage = "down";
    if (k > 160 && squatStage === "down" && Date.now() - lastRepTime > MIN_REP_INTERVAL) {
      squatStage = "up";
      repCount++;
      repCountEl.textContent = repCount;
      lastRepTime = Date.now();
      msg = "+1 присед 💪";
      color = "#00ff00";
    } else {
      msg = k < 100 ? "Отличная глубина 🔥" : "Приседайте глубже";
    }
  }

  if (ex === "plank") {
    if (!plankStart) plankStart = Date.now();
    const sec = Math.floor((Date.now() - plankStart) / 1000);
    timerEl.textContent = sec;
    msg = `Планка ${sec} сек`;
    color = "#00ff00";
  }

  if (ex === "none") {
    msg = "Стойка. Начните упражнение.";
  }

  feedbackEl.textContent = msg;
  feedbackEl.style.color = color;
}

/* =========================
   VIDEO LOOP
========================= */
function runVideo() {
  if (!AppState.cameraRunning) return;

  const res = poseLandmarker.detectForVideo(video, performance.now());
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (res.landmarks?.length) {
    const lm = res.landmarks[0];
    const ex = AppState.exercise === "auto" ? detectExercise(lm) : AppState.exercise;

    exerciseNameEl.textContent = EXERCISE_NAMES[ex];
    handleExercise(ex, lm);

    if (AppState.drawingEnabled) {
      const d = new DrawingUtils(ctx);
      d.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS);
      d.drawLandmarks(lm);
    }
  }

  requestAnimationFrame(runVideo);
}

/* =========================
   CAMERA CONTROLS
========================= */
startBtn.onclick = async () => {
  if (AppState.cameraRunning) return;
  await loadModel();

  stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await video.play();

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  AppState.cameraRunning = true;
  AppState.mode = "video";
  resetCounters();
  updateControls();
  runVideo();
};

stopBtn.onclick = () => {
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;

  AppState.cameraRunning = false;
  AppState.mode = "idle";

  updateControls();
  renderIdle();
};

resetBtn.onclick = resetCounters;
