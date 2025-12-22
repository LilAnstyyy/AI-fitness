import { FilesetResolver, PoseLandmarker, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById('webcam');
const canvas = document.getElementById('output_canvas');
const ctx = canvas.getContext('2d');
const repCountEl = document.getElementById('repCount');
const timerEl = document.getElementById('timer');
const feedbackEl = document.getElementById('feedback');
const exerciseNameEl = document.getElementById('exerciseName');
const photoPreview = document.getElementById('photoPreview');
const photoFeedbackEl = document.getElementById('photoFeedback');
const photoExerciseNameEl = document.getElementById('photoExerciseName');

let poseLandmarker = null;
let isCameraRunning = false;
let stream = null;
let repCount = 0;
let plankStartTime = 0;
let currentExercise = 'auto';
let squatStage = 'up';
let lungeStage = 'up';
let lastRepTime = 0;
const minRepInterval = 800;

const EXERCISE_NAMES = {
  squats: 'Приседания',
  lunges: 'Выпады (болгарские)',
  plank: 'Планка'
};

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
    numPoses: 1,
    minPoseDetectionConfidence: 0.6,
    minPosePresenceConfidence: 0.6,
    minTrackingConfidence: 0.6
  });

  feedbackEl.textContent = "Модель загружена. Включите камеру или загрузите фото.";
}

function calculateAngle(a, b, c) {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs(radians * 180.0 / Math.PI);
  if (angle > 180) angle = 360 - angle;
  return angle;
}

function isBodyHorizontal(landmarks) {
  const lShoulder = landmarks[11];
  const rShoulder = landmarks[12];
  const lHip = landmarks[23];
  const rHip = landmarks[24];
  const shoulderY = (lShoulder.y + rShoulder.y) / 2;
  const hipY = (lHip.y + rHip.y) / 2;
  return Math.abs(shoulderY - hipY) < 0.15 && shoulderY > 0.2 && hipY > 0.2;
}

function detectExercise(landmarks) {
  // Проверка видимости ключевых точек ног
  const legVisibility = [landmarks[23], landmarks[24], landmarks[25], landmarks[26], landmarks[27], landmarks[28]]
    .reduce((sum, lm) => sum + (lm.visibility || 0), 0) / 6;
  if (legVisibility < 0.5) return 'none';

  const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
  const rHip = landmarks[24], rKnee = landmarks[26], rAnkle = landmarks[28];
  const lShoulder = landmarks[11];

  const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
  const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
  const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;
  const kneeDiff = Math.abs(leftKneeAngle - rightKneeAngle);
  const bodyLineAngle = calculateAngle(lShoulder, lHip, lAnkle);

  // Приседания или выпады — только если колени согнуты и ноги видны
  if (avgKneeAngle < 150 && legVisibility > 0.7) {
    if (kneeDiff > 40) return 'lunges';
    return 'squats';
  }

  // Планка — только если тело горизонтальное и ключевые точки видны
  if (avgKneeAngle > 160 && bodyLineAngle > 160 && isBodyHorizontal(landmarks)) {
    return 'plank';
  }

  return 'none';
}

function giveFeedback(exercise, landmarks) {
  if (exercise === 'none') {
    feedbackEl.style.color = '#ffcc00';
    return 'Стартовая позиция. Начните упражнение!';
  }

  const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
  const rHip = landmarks[24], rKnee = landmarks[26];
  const lShoulder = landmarks[11];

  let msg = '';
  let color = '#ff4757';

  if (exercise === 'squats') {
    const avgKneeAngle = (calculateAngle(lHip, lKnee, lAnkle) + calculateAngle(rHip, rKnee, rAnkle)) / 2;
    const hipAngle = calculateAngle(lShoulder, lHip, lKnee);

    if (avgKneeAngle < 100 && hipAngle > 140) {
      msg = 'Отлично! Глубокий присед, спина прямая 🔥';
      color = '#00ff00';
    } else if (avgKneeAngle < 100) {
      msg = 'Глубоко, но спина наклоняется — держите грудь вверх!';
    } else {
      msg = 'Приседайте глубже (колени под ~90°)';
    }

    if (avgKneeAngle < 95 && squatStage === 'up') squatStage = 'down';
    if (avgKneeAngle > 155 && squatStage === 'down') {
      if (Date.now() - lastRepTime > minRepInterval) {
        squatStage = 'up';
        repCount++;
        repCountEl.textContent = repCount;
        lastRepTime = Date.now();
        msg = 'Супер! +1 повторение 💪';
        color = '#00ff00';
      }
    }

  } else if (exercise === 'lunges') {
    const frontAngle = Math.min(calculateAngle(lHip, lKnee, lAnkle), calculateAngle(rHip, rKnee, rAnkle));

    if (frontAngle > 80 && frontAngle < 100) {
      msg = 'Идеально! Переднее колено под 90° 👌';
      color = '#00ff00';
    } else if (frontAngle < 80) {
      msg = 'Согните переднюю ногу сильнее';
    } else {
      msg = 'Не переразгибайте переднее колено';
    }

    if (frontAngle < 85 && lungeStage === 'up') lungeStage = 'down';
    if (frontAngle > 140 && lungeStage === 'down') {
      if (Date.now() - lastRepTime > minRepInterval) {
        lungeStage = 'up';
        repCount++;
        repCountEl.textContent = repCount;
        lastRepTime = Date.now();
      }
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
      plankStartTime = 0;
      timerEl.textContent = '0';
    }
  }

  feedbackEl.style.color = color;
  return msg;
}

function processFrame(results) {
  // Ключевой фикс: всегда рисуем текущее видео
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (results.landmarks && results.landmarks.length > 0) {
    const landmarks = results.landmarks[0];
    const drawingUtils = new DrawingUtils(ctx);
    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
    drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', radius: 6 });

    let detected = currentExercise === 'auto' ? detectExercise(landmarks) : currentExercise;
    if (detected === 'none' && currentExercise !== 'auto') detected = currentExercise;

    exerciseNameEl.textContent = EXERCISE_NAMES[detected] || 'Не определено';
    feedbackEl.textContent = giveFeedback(detected, landmarks);
  } else {
    feedbackEl.textContent = 'Поза не обнаружена. Встаньте полностью в кадр.';
    feedbackEl.style.color = '#ff4757';
  }
}

function runDetection() {
  if (!isCameraRunning || !poseLandmarker) return;
  const results = poseLandmarker.detectForVideo(video, performance.now());
  processFrame(results);
  requestAnimationFrame(runDetection);
}

// Камера
document.getElementById('startButton').addEventListener('click', async () => {
  if (isCameraRunning) return;

  if (!poseLandmarker) await initPoseLandmarker();

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      video.play();
      isCameraRunning = true;
      document.getElementById('startButton').disabled = true;
      document.getElementById('stopButton').disabled = false;
      feedbackEl.textContent = "Камера включена. Начните упражнение!";
      runDetection();
    };
  } catch (err) {
    feedbackEl.textContent = "Ошибка камеры: " + err.message;
    feedbackEl.style.color = '#ff4757';
  }
});

document.getElementById('stopButton').addEventListener('click', () => {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  isCameraRunning = false;
  document.getElementById('startButton').disabled = false;
  document.getElementById('stopButton').disabled = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  feedbackEl.textContent = "Камера выключена";
});

document.getElementById('resetButton').addEventListener('click', () => {
  repCount = 0;
  plankStartTime = 0;
  squatStage = 'up';
  lungeStage = 'up';
  lastRepTime = 0;
  repCountEl.textContent = '0';
  timerEl.textContent = '0';
});

// Выбор упражнения
document.querySelectorAll('.exercise-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.exercise-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentExercise = btn.dataset.exercise;
  });
});

// Фото
const photoUpload = document.getElementById('photoUpload');
const photoUploadArea = document.getElementById('photoUploadArea');
const analyzeBtn = document.getElementById('analyzePhotoButton');
const clearBtn = document.getElementById('clearPhotoButton');

photoUploadArea.addEventListener('click', () => photoUpload.click());

photoUpload.addEventListener('change', () => {
  if (photoUpload.files && photoUpload.files[0]) {
    photoPreview.src = URL.createObjectURL(photoUpload.files[0]);
    photoPreview.style.display = 'block';
    analyzeBtn.disabled = false;
    clearBtn.disabled = false;
  }
});

analyzeBtn.addEventListener('click', async () => {
  if (!photoPreview.src || !poseLandmarker) {
    photoFeedbackEl.textContent = "Загрузите фото и подождите загрузки модели.";
    return;
  }

  const img = new Image();
  img.src = photoPreview.src;
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    const mpImage = new mp.Image(img, mp.ImageFormat.SRGB);
    const results = poseLandmarker.detect(mpImage);

    if (results.landmarks && results.landmarks.length > 0) {
      const landmarks = results.landmarks[0];
      const drawingUtils = new DrawingUtils(ctx);
      drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
      drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', radius: 6 });

      const detected = currentExercise === 'auto' ? detectExercise(landmarks) : currentExercise;
      photoExerciseNameEl.textContent = EXERCISE_NAMES[detected] || 'Не определено';
      photoFeedbackEl.textContent = giveFeedback(detected, landmarks);
      photoFeedbackEl.style.color = feedbackEl.style.color;
    } else {
      photoFeedbackEl.textContent = 'Поза не обнаружена на фото. Попробуйте фронтальный или полу-боковой ракурс.';
      photoFeedbackEl.style.color = '#ff4757';
    }
  };
});

clearBtn.addEventListener('click', () => {
  photoPreview.src = '';
  photoPreview.style.display = 'none';
  photoUpload.value = '';
  photoExerciseNameEl.textContent = '—';
  photoFeedbackEl.textContent = 'Загрузите фото для анализа';
  analyzeBtn.disabled = true;
  clearBtn.disabled = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// Инициализация
document.addEventListener('DOMContentLoaded', () => {
  feedbackEl.textContent = "Загрузка модели ИИ...";
  initPoseLandmarker();
});