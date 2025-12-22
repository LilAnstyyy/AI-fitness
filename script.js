import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ===== DOM =====
const video = document.getElementById("webcam");
const canvas = document.getElementById("output_canvas");
const ctx = canvas.getContext("2d");

const repCountEl = document.getElementById("repCount");
const timerEl = document.getElementById("timer");
const feedbackEl = document.getElementById("feedback");
const exerciseNameEl = document.getElementById("exerciseName");

// ===== STATE =====
let poseLandmarker = null;
let currentExercise = "none";
let previousExercise = "none";

let repCount = 0;
let squatStage = null;
let lungeStage = null;
let plankStartTime = 0;

// ===== INIT =====
async function initPoseLandmarker() {
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

  feedbackEl.textContent = "Модель загружена. Начните тренировку или загрузите фото.";
  feedbackEl.style.color = "#ffd93d";
}

// ===== UTILS =====
function calculateAngle(a, b, c) {
  const radians =
    Math.atan2(c.y - b.y, c.x - b.x) -
    Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180) / Math.PI);
  if (angle > 180) angle = 360 - angle;
  return angle;
}

function isBodyHorizontal(landmarks) {
  const shoulderY = (landmarks[11].y + landmarks[12].y) / 2;
  const hipY = (landmarks[23].y + landmarks[24].y) / 2;
  return Math.abs(shoulderY - hipY) < 0.12;
}

// ===== EXERCISE DETECTION =====
function detectExercise(landmarks) {
  const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
  const rHip = landmarks[24], rKnee = landmarks[26], rAnkle = landmarks[28];
  const lShoulder = landmarks[11];

  const leftKnee = calculateAngle(lHip, lKnee, lAnkle);
  const rightKnee = calculateAngle(rHip, rKnee, rAnkle);
  const avgKnee = (leftKnee + rightKnee) / 2;

  const hipAngle = calculateAngle(lShoulder, lHip, lKnee);

  // Планка
  if (avgKnee > 160 && isBodyHorizontal(landmarks)) {
    return "plank";
  }

  // Присед
  if (avgKnee < 130 && hipAngle < 140) {
    return "squats";
  }

  // Выпады
  if (
    (leftKnee < 120 && rightKnee > 150) ||
    (rightKnee < 120 && leftKnee > 150)
  ) {
    return "lunges";
  }

  return "none";
}

// ===== FEEDBACK + COUNTERS =====
function giveFeedback(exercise, landmarks) {
  feedbackEl.style.color = "#ffd93d";

  if (exercise === "none") {
    return "Встаньте в стартовую позицию и начните упражнение";
  }

  const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
  const rHip = landmarks[24], rKnee = landmarks[26];
  const lShoulder = landmarks[11];

  // ===== SQUATS =====
  if (exercise === "squats") {
    const kneeAngle =
      (calculateAngle(lHip, lKnee, lAnkle) +
        calculateAngle(rHip, rKnee, lAnkle)) /
      2;

    if (kneeAngle < 95 && squatStage !== "down") {
      squatStage = "down";
    }

    if (kneeAngle > 160 && squatStage === "down") {
      squatStage = "up";
      repCount++;
      repCountEl.textContent = repCount;
    }

    if (kneeAngle < 100) {
      feedbackEl.style.color = "#00ff00";
      return "Отличный глубокий присед 🔥";
    } else {
      feedbackEl.style.color = "#ff4757";
      return "Приседайте глубже (колени ~90°)";
    }
  }

  // ===== LUNGES =====
  if (exercise === "lunges") {
    const left = calculateAngle(lHip, lKnee, lAnkle);
    const right = calculateAngle(rHip, rKnee, lAnkle);
    const front = Math.min(left, right);

    if (front < 90 && lungeStage !== "down") {
      lungeStage = "down";
    }

    if (front > 150 && lungeStage === "down") {
      lungeStage = "up";
      repCount++;
      repCountEl.textContent = repCount;
    }

    if (front > 80 && front < 100) {
      feedbackEl.style.color = "#00ff00";
      return "Идеальный выпад 👌";
    } else {
      feedbackEl.style.color = "#ff4757";
      return "Переднее колено должно быть ~90°";
    }
  }

  // ===== PLANK =====
  if (exercise === "plank") {
    const lineAngle = calculateAngle(lShoulder, lHip, lAnkle);

    if (lineAngle > 170) {
      if (!plankStartTime) plankStartTime = Date.now();
      timerEl.textContent = Math.floor(
        (Date.now() - plankStartTime) / 1000
      );
      feedbackEl.style.color = "#00ff00";
      return "Тело ровное, держите планку 💪";
    } else {
      plankStartTime = 0;
      timerEl.textContent = "0";
      feedbackEl.style.color = "#ff4757";
      return "Выпрямите спину и таз";
    }
  }
}

// ===== PROCESS RESULTS =====
function processResults(results, source) {
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);

  if (!results.landmarks || results.landmarks.length === 0) {
    currentExercise = "none";
    exerciseNameEl.textContent = "Нет позы";
    feedbackEl.textContent = "Поза не обнаружена";
    feedbackEl.style.color = "#ff4757";
    return;
  }

  const landmarks = results.landmarks[0];
  const drawingUtils = new DrawingUtils(ctx);

  drawingUtils.drawConnectors(
    landmarks,
    PoseLandmarker.POSE_CONNECTIONS,
    { color: "#00ff00", lineWidth: 4 }
  );
  drawingUtils.drawLandmarks(landmarks, {
    color: "#ff0000",
    radius: 5
  });

  currentExercise = detectExercise(landmarks);

  if (currentExercise !== previousExercise) {
    previousExercise = currentExercise;
    repCount = 0;
    repCountEl.textContent = "0";
    timerEl.textContent = "0";
    squatStage = null;
    lungeStage = null;
    plankStartTime = 0;

    const names = {
      squats: "Приседания",
      lunges: "Выпады",
      plank: "Планка"
    };
    exerciseNameEl.textContent = names[currentExercise] || "Старт";
  }

  feedbackEl.textContent = giveFeedback(currentExercise, landmarks);
}

// ===== VIDEO LOOP =====
function runVideoDetection() {
  if (!poseLandmarker) return;
  const results = poseLandmarker.detectForVideo(video, performance.now());
  processResults(results, video);
  requestAnimationFrame(runVideoDetection);
}

// ===== CAMERA =====
document.getElementById("startButton").addEventListener("click", async () => {
  if (!poseLandmarker) await initPoseLandmarker();

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" }
  });

  video.srcObject = stream;
  video.play();

  video.onloadedmetadata = () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    runVideoDetection();
  };
});

// ===== PHOTO =====
document
  .getElementById("analyzePhotoButton")
  .addEventListener("click", async () => {
    const input = document.getElementById("photoUpload");
    if (!input.files.length) {
      feedbackEl.textContent = "Выберите фото";
      feedbackEl.style.color = "#ff4757";
      return;
    }

    if (!poseLandmarker) await initPoseLandmarker();

    const img = new Image();
    img.src = URL.createObjectURL(input.files[0]);

    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;

      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(
        0,
        0,
        canvas.width,
        canvas.height
      );

      const results = poseLandmarker.detect(imageData);
      processResults(results, img);
    };
  });
