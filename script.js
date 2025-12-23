import { FilesetResolver, PoseLandmarker, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById('webcam');
const canvas = document.getElementById('output_canvas');
const ctx = canvas.getContext('2d');
const repCountEl = document.getElementById('repCount');
const timerEl = document.getElementById('timer');
const feedbackEl = document.getElementById('feedback');
const exerciseNameEl = document.getElementById('exerciseName');

let poseLandmarker = null;
let repCount = 0;
let plankStartTime = 0;
let currentExercise = 'none';
let previousExercise = 'none';
let squatStage = null;
let lungeStage = null;

async function initPoseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numPoses: 1
  });

  feedbackEl.textContent = "Камера готова! Встаньте в кадр полностью.";
  exerciseNameEl.textContent = "Определение упражнения...";
}

function calculateAngle(a, b, c) {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs(radians * 180.0 / Math.PI);
  if (angle > 180) angle = 360 - angle;
  return angle;
}

function detectExercise(landmarks) {
  const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
  const rHip = landmarks[24], rKnee = landmarks[26], rAnkle = landmarks[28];
  const lShoulder = landmarks[11], rShoulder = landmarks[12];

  const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
  const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
  const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;
  const kneeDiff = Math.abs(leftKneeAngle - rightKneeAngle);

  const bodyLineAngle = calculateAngle(lShoulder, lHip, lAnkle);

  if (bodyLineAngle > 165 && avgKneeAngle > 150) {
    return 'plank';
  } else if (kneeDiff > 30 && (leftKneeAngle < 120 || rightKneeAngle < 120)) {
    return 'lunges';
  } else if (avgKneeAngle < 140) {
    return 'squats';
  }
  return 'none';
}

function giveFeedback(exercise, landmarks) {
  if (exercise === 'none') {
    feedbackEl.style.color = '#ffd93d';
    return 'Встаньте в кадр и начните упражнение';
  }

  const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
  const rHip = landmarks[24], rKnee = landmarks[26], rAnkle = landmarks[28];
  const lShoulder = landmarks[11];

  let msg = '';
  let color = '#ff4757';

  if (exercise === 'squats') {
    const kneeAngle = (calculateAngle(lHip, lKnee, lAnkle) + calculateAngle(rHip, rKnee, rAnkle)) / 2;
    const hipAngle = calculateAngle(lShoulder, lHip, lKnee);

    if (kneeAngle < 95) {
      if (hipAngle > 140) {
        msg = 'Отлично! Глубокий присед, спина прямая 🔥';
        color = '#00ff00';
      } else {
        msg = 'Спина наклоняется — держите грудь вверх!';
      }
    } else {
      msg = 'Приседайте глубже (колени под ~90°)';
    }

    if (kneeAngle < 95) squatStage = 'down';
    if (kneeAngle > 155 && squatStage === 'down') {
      squatStage = 'up';
      repCount++;
      repCountEl.textContent = repCount;
      msg = 'Отличное повторение! +1 💪';
      color = '#00ff00';
    }
  } else if (exercise === 'lunges') {
    const leftAngle = calculateAngle(lHip, lKnee, lAnkle);
    const rightAngle = calculateAngle(rHip, rKnee, rAnkle);
    const frontAngle = Math.min(leftAngle, rightAngle);

    if (frontAngle > 80 && frontAngle < 100) {
      msg = 'Идеально! Переднее колено под 90° 👌';
      color = '#00ff00';
    } else if (frontAngle < 80) {
      msg = 'Согните переднюю ногу сильнее';
    } else {
      msg = 'Не переразгибайте переднее колено';
    }

    if (frontAngle < 85) lungeStage = 'down';
    if (frontAngle > 140 && lungeStage === 'down') {
      lungeStage = 'up';
      repCount++;
      repCountEl.textContent = repCount;
    }
  } else if (exercise === 'plank') {
    const lineAngle = calculateAngle(lShoulder, lHip, lAnkle);
    if (lineAngle > 170) {
      if (plankStartTime === 0) plankStartTime = Date.now();
      const seconds = Math.floor((Date.now() - plankStartTime) / 1000);
      timerEl.textContent = seconds;
      msg = 'Держите! Тело прямое как доска 💪';
      color = '#00ff00';
    } else {
      msg = 'Провисает спина или таз — выпрямитесь!';
      if (plankStartTime !== 0) {
        plankStartTime = 0;
        timerEl.textContent = '0';
      }
    }
  }

  feedbackEl.style.color = color;
  return msg;
}

function runDetection() {
  if (!poseLandmarker) return;

  const results = poseLandmarker.detectForVideo(video, performance.now());

  // Отражаем видео по горизонтали (зеркало)
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  if (results.landmarks && results.landmarks.length > 0) {
    const landmarks = results.landmarks[0];

    // Отражаем скелет, чтобы он совпадал с зеркальным изображением
    ctx.save();
    ctx.scale(-1, 1);
    const drawingUtils = new DrawingUtils(ctx);
    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
    drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', radius: 6 });
    ctx.restore();

    const detected = detectExercise(landmarks);
    if (detected !== 'none') {
      currentExercise = detected;
    }

    if (currentExercise !== previousExercise) {
      previousExercise = currentExercise;
      repCount = 0;
      repCountEl.textContent = '0';
      plankStartTime = 0;
      timerEl.textContent = '0';
      squatStage = null;
      lungeStage = null;

      const names = { squats: 'Приседания', lunges: 'Выпады (болгарские)', plank: 'Планка' };
      exerciseNameEl.textContent = names[currentExercise] || 'Определение...';
    }

    const feedback = giveFeedback(currentExercise, landmarks);
    feedbackEl.textContent = feedback;
  } else {
    feedbackEl.textContent = 'Поза не обнаружена. Встаньте полностью в кадр.';
    exerciseNameEl.textContent = '—';
  }

  requestAnimationFrame(runDetection);
}

// Запуск камеры
document.getElementById('startButton').addEventListener('click', async () => {
  if (!poseLandmarker) await initPoseLandmarker();

  navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } })
    .then(stream => {
      video.srcObject = stream;
      video.play();

      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        runDetection();
      };
    })
    .catch(err => {
      feedbackEl.textContent = "Ошибка камеры: " + err.message;
    });
});