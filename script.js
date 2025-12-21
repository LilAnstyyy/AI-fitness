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

  feedbackEl.textContent = "Модель загружена! Начните тренировку или загрузите фото.";
  feedbackEl.style.color = '#ffd93d';
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
  return Math.abs(shoulderY - hipY) < 0.15;
}

function detectExercise(landmarks) {
  const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
  const rHip = landmarks[24], rKnee = landmarks[26], rAnkle = landmarks[28];
  const lShoulder = landmarks[11];

  const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
  const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
  const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;
  const kneeDiff = Math.abs(leftKneeAngle - rightKneeAngle);

  const bodyLineAngle = calculateAngle(lShoulder, lHip, lAnkle);

  if (avgKneeAngle < 150) {
    if (kneeDiff > 40) return 'lunges';
    return 'squats';
  }

  if (avgKneeAngle > 160 && bodyLineAngle > 160 && isBodyHorizontal(landmarks)) {
    return 'plank';
  }

  return 'none';
}

function giveFeedback(exercise, landmarks) {
  feedbackEl.style.color = '#ffd93d'; // Жёлтый по умолчанию

  if (exercise === 'none') {
    return 'Стартовая позиция. Начните упражнение!';
  }

  const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
  const rHip = landmarks[24], rKnee = landmarks[26];
  const lShoulder = landmarks[11];

  let msg = '';

  if (exercise === 'squats') {
    const avgKneeAngle = (calculateAngle(lHip, lKnee, lAnkle) + calculateAngle(rHip, rKnee, rAnkle)) / 2;
    const hipAngle = calculateAngle(lShoulder, lHip, lKnee);

    if (avgKneeAngle < 100 && hipAngle > 140) {
      msg = 'Отлично! Глубокий присед, спина прямая 🔥';
      feedbackEl.style.color = '#00ff00';
    } else if (avgKneeAngle < 100) {
      msg = 'Глубоко, но спина наклоняется — держите грудь вверх!';
      feedbackEl.style.color = '#ff4757';
    } else {
      msg = 'Приседайте глубже (колени под ~90°)';
      feedbackEl.style.color = '#ff4757';
    }

    if (avgKneeAngle < 95) squatStage = 'down';
    if (avgKneeAngle > 155 && squatStage === 'down') {
      squatStage = 'up';
      repCount++;
      repCountEl.textContent = repCount;
      msg = 'Супер! +1 повторение 💪';
      feedbackEl.style.color = '#00ff00';
    }

  } else if (exercise === 'lunges') {
    const leftAngle = calculateAngle(lHip, lKnee, lAnkle);
    const rightAngle = calculateAngle(rHip, rKnee, rAnkle);
    const frontAngle = Math.min(leftAngle, rightAngle);

    if (frontAngle > 80 && frontAngle < 100) {
      msg = 'Идеально! Переднее колено под 90° 👌';
      feedbackEl.style.color = '#00ff00';
    } else if (frontAngle < 80) {
      msg = 'Согните переднюю ногу сильнее';
      feedbackEl.style.color = '#ff4757';
    } else {
      msg = 'Не переразгибайте переднее колено';
      feedbackEl.style.color = '#ff4757';
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
      feedbackEl.style.color = '#00ff00';
    } else {
      msg = 'Провисает спина или таз — выпрямитесь!';
      feedbackEl.style.color = '#ff4757';
      plankStartTime = 0;
      timerEl.textContent = '0';
    }
  }

  return msg;
}

function processResults(results, sourceImage) {
  ctx.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);

  if (results.landmarks && results.landmarks.length > 0) {
    const landmarks = results.landmarks[0];
    const drawingUtils = new DrawingUtils(ctx);
    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
    drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', radius: 6 });

    const detected = detectExercise(landmarks);
    if (detected !== 'none') currentExercise = detected;

    if (currentExercise !== previousExercise) {
      previousExercise = currentExercise;
      repCount = 0;
      repCountEl.textContent = '0';
      plankStartTime = 0;
      timerEl.textContent = '0';
      squatStage = null;
      lungeStage = null;

      const names = { squats: 'Приседания', lunges: 'Выпады (болгарские)', plank: 'Планка' };
      exerciseNameEl.textContent = names[currentExercise] || 'Стартовая позиция';
    }

    feedbackEl.textContent = giveFeedback(currentExercise, landmarks);
  } else {
    feedbackEl.textContent = 'Поза не обнаружена. Встаньте в полный рост, хорошее освещение, полу-боковой ракурс.';
    feedbackEl.style.color = '#ff4757';
  }
}

function runVideoDetection() {
  if (!poseLandmarker) return;
  const results = poseLandmarker.detectForVideo(video, performance.now());
  processResults(results, video);
  requestAnimationFrame(runVideoDetection);
}

// Камера
document.getElementById('startButton').addEventListener('click', async () => {
  if (!poseLandmarker) await initPoseLandmarker();

  navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } })
    .then(stream => {
      video.srcObject = stream;
      video.play();
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        runVideoDetection();
      };
    })
    .catch(err => {
      feedbackEl.textContent = "Ошибка камеры: " + err.message;
      feedbackEl.style.color = '#ff4757';
    });
});

// Фото
document.getElementById('analyzePhotoButton').addEventListener('click', async () => {
  const fileInput = document.getElementById('photoUpload');
  if (!fileInput.files || fileInput.files.length === 0) {
    feedbackEl.textContent = 'Выберите фото!';
    feedbackEl.style.color = '#ff4757';
    return;
  }

  if (!poseLandmarker) await initPoseLandmarker();

  const file = fileInput.files[0];
  const img = new Image();
  img.src = URL.createObjectURL(file);

  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;

    const mpImage = new mp.Image(img, mp.ImageFormat.SRGB);
    const results = poseLandmarker.detect(mpImage);

    processResults(results, img);
  };
});