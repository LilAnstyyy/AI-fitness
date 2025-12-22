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
let pushupStage = 'up';
let lastRepTime = 0;
const minRepInterval = 800;
let isDrawingEnabled = true;

const EXERCISE_NAMES = {
  squats: 'Приседания',
  lunges: 'Выпады',
  plank: 'Планка',
  pushups: 'Отжимания',
  none: 'Стойка'
};

async function loadModel() {
  if (poseLandmarker) return;

  feedbackEl.textContent = "Загрузка модели ИИ... (один раз)";
  photoFeedbackEl.textContent = "Загрузка модели...";

  try {
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

    feedbackEl.textContent = "Модель загружена! Готово к работе.";
    photoFeedbackEl.textContent = "Модель загружена. Загрузите фото.";
  } catch (err) {
    feedbackEl.textContent = "Ошибка загрузки модели. Проверьте интернет.";
    feedbackEl.style.color = '#ff4757';
    console.error(err);
  }
}

function calculateAngle(a, b, c) {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs(radians * 180.0 / Math.PI);
  if (angle > 180) angle = 360 - angle;
  return Math.round(angle);
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
  const visibilityAvg = [11,12,23,24,25,26,27,28].reduce((s, i) => s + (landmarks[i]?.visibility || 0), 0) / 8;
  if (visibilityAvg < 0.5) return 'none';

  const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
  const rHip = landmarks[24], rKnee = landmarks[26], rAnkle = landmarks[28];
  const lShoulder = landmarks[11], rShoulder = landmarks[12];
  const lElbow = landmarks[13], rElbow = landmarks[14];
  const lWrist = landmarks[15], rWrist = landmarks[16];

  const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
  const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
  const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;
  const kneeDiff = Math.abs(leftKneeAngle - rightKneeAngle);

  const leftElbowAngle = calculateAngle(lShoulder, lElbow, lWrist);
  const rightElbowAngle = calculateAngle(rShoulder, rElbow, rWrist);
  const avgElbowAngle = (leftElbowAngle + rightElbowAngle) / 2;

  const bodyLineAngle = calculateAngle(lShoulder, lHip, lAnkle);

  if (avgElbowAngle < 130 && isBodyHorizontal(landmarks) && bodyLineAngle > 160) {
    return 'pushups';
  }

  if (avgKneeAngle < 140) {
    if (kneeDiff > 35) return 'lunges';
    return 'squats';
  }

  if (avgKneeAngle > 160 && bodyLineAngle > 170 && isBodyHorizontal(landmarks)) {
    return 'plank';
  }

  return 'none';
}

function getDetailedAdvice(exercise, landmarks) {
  const advice = [];

  if (exercise === 'squats') {
    const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
    const rHip = landmarks[24], rKnee = landmarks[26], rAnkle = landmarks[28];
    const lShoulder = landmarks[11];

    const avgKneeAngle = (calculateAngle(lHip, lKnee, lAnkle) + calculateAngle(rHip, rKnee, rAnkle)) / 2;
    const hipAngle = calculateAngle(lShoulder, lHip, lKnee);

    if (avgKneeAngle > 120) advice.push("• Приседайте глубже (цель — 90° в коленях)");
    if (hipAngle < 140) advice.push("• Держите спину прямой, грудь вперёд");
    if (Math.abs(calculateAngle(lHip, lKnee, lAnkle) - calculateAngle(rHip, rKnee, rAnkle)) > 20) advice.push("• Равномерно распределяйте вес");

    if (advice.length === 0) advice.push("• Отличная техника! Продолжайте 🔥");

  } else if (exercise === 'lunges') {
    const leftAngle = calculateAngle(landmarks[23], landmarks[25], landmarks[27]);
    const rightAngle = calculateAngle(landmarks[24], landmarks[26], landmarks[28]);
    const frontAngle = Math.min(leftAngle, rightAngle);

    if (frontAngle > 100) advice.push("• Согните переднее колено сильнее (цель 90°)");
    if (frontAngle < 80) advice.push("• Не опускайтесь слишком низко");
    if (Math.abs(landmarks[23].y - landmarks[24].y) > 0.1) advice.push("• Держите таз ровно");

    if (advice.length === 0) advice.push("• Идеальный выпад! Колено не выходит за носок 👌");

  } else if (exercise === 'plank') {
    const lineAngle = calculateAngle(landmarks[11], landmarks[23], landmarks[27]);

    if (lineAngle < 170) advice.push("• Подтяните живот, выпрямите тело");
    if (landmarks[23].y < landmarks[11].y) advice.push("• Опустите таз");

    if (advice.length === 0) advice.push("• Отличная планка! Держитесь 💪");

  } else if (exercise === 'pushups') {
    const avgElbowAngle = (calculateAngle(landmarks[11], landmarks[13], landmarks[15]) + 
                           calculateAngle(landmarks[12], landmarks[14], landmarks[16])) / 2;
    const bodyLineAngle = calculateAngle(landmarks[11], landmarks[23], landmarks[27]);

    if (avgElbowAngle > 130) advice.push("• Опуститесь глубже");
    if (bodyLineAngle < 170) advice.push("• Держите тело прямым");

    if (advice.length === 0) advice.push("• Отличная техника отжиманий!");

  }

  return advice.join('\n');
}

function giveFeedback(exercise, landmarks) {
  if (exercise === 'none') {
    return {
      message: 'Стартовая позиция. Начните упражнение!',
      color: '#ffcc00',
      advice: 'Советы:\n• Встаньте боком к камере\n• Все тело должно быть в кадре\n• Хорошее освещение'
    };
  }

  let message = '';
  let color = '#ff4757';
  let advice = getDetailedAdvice(exercise, landmarks);

  if (exercise === 'squats') {
    const avgKneeAngle = (calculateAngle(landmarks[23], landmarks[25], landmarks[27]) + 
                          calculateAngle(landmarks[24], landmarks[26], landmarks[28])) / 2;

    if (avgKneeAngle < 100) {
      message = 'Отличный присед! 🔥';
      color = '#00ff00';
    } else if (avgKneeAngle < 130) {
      message = 'Хорошо, можно глубже';
      color = '#ffcc00';
    } else {
      message = 'Начните приседание';
    }

    if (avgKneeAngle < 95 && squatStage === 'up') squatStage = 'down';
    if (avgKneeAngle > 155 && squatStage === 'down') {
      if (Date.now() - lastRepTime > minRepInterval) {
        squatStage = 'up';
        repCount++;
        repCountEl.textContent = repCount;
        lastRepTime = Date.now();
        message = 'Отлично! +1 повторение 💪';
        color = '#00ff00';
      }
    }

  } else if (exercise === 'lunges') {
    const frontAngle = Math.min(calculateAngle(landmarks[23], landmarks[25], landmarks[27]), 
                                calculateAngle(landmarks[24], landmarks[26], landmarks[28]));

    if (frontAngle > 80 && frontAngle < 100) {
      message = 'Идеально! 90° в переднем колене 👌';
      color = '#00ff00';
    } else {
      message = frontAngle < 80 ? 'Согните сильнее' : 'Не переразгибайте';
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
    const lineAngle = calculateAngle(landmarks[11], landmarks[23], landmarks[27]);
    if (lineAngle > 170) {
      if (plankStartTime === 0) plankStartTime = Date.now();
      const seconds = Math.floor((Date.now() - plankStartTime) / 1000);
      timerEl.textContent = seconds;
      message = `Держите! ${seconds} сек. 💪`;
      color = '#00ff00';
    } else {
      message = 'Выпрямите тело!';
      plankStartTime = 0;
      timerEl.textContent = '0';
    }

  } else if (exercise === 'pushups') {
    const avgElbowAngle = (calculateAngle(landmarks[11], landmarks[13], landmarks[15]) + 
                           calculateAngle(landmarks[12], landmarks[14], landmarks[16])) / 2;

    if (avgElbowAngle < 100) {
      message = 'Отлично! Полная амплитуда 💪';
      color = '#00ff00';
    } else {
      message = avgElbowAngle < 130 ? 'Опуститесь ниже' : 'Начните отжимание';
    }

    if (avgElbowAngle < 100 && pushupStage === 'up') pushupStage = 'down';
    if (avgElbowAngle > 150 && pushupStage === 'down') {
      if (Date.now() - lastRepTime > minRepInterval) {
        pushupStage = 'up';
        repCount++;
        repCountEl.textContent = repCount;
        lastRepTime = Date.now();
      }
    }
  }

  return { message, color, advice };
}

function processFrame(results) {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (results.landmarks && results.landmarks.length > 0) {
    const landmarks = results.landmarks[0];

    if (isDrawingEnabled) {
      const drawingUtils = new DrawingUtils(ctx);
      drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
      drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', radius: 6 });
    }

    let detected = currentExercise === 'auto' ? detectExercise(landmarks) : currentExercise;
    if (detected === 'none' && currentExercise !== 'auto') detected = currentExercise;

    exerciseNameEl.textContent = EXERCISE_NAMES[detected] || 'Определение...';

    const fb = giveFeedback(detected, landmarks);
    feedbackEl.innerHTML = `<strong>${fb.message}</strong><br><small style="color: #aaa; white-space: pre-line;">${fb.advice}</small>`;
    feedbackEl.style.color = fb.color;
  } else {
    feedbackEl.textContent = 'Поза не обнаружена. Встаньте боком, всё тело в кадре.';
    feedbackEl.style.color = '#ff4757';
    exerciseNameEl.textContent = '—';
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

  await loadModel();

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    video.srcObject = stream;

    video.onloadedmetadata = () => {
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      video.play();
      isCameraRunning = true;
      document.getElementById('startButton').disabled = true;
      document.getElementById('stopButton').disabled = false;
      feedbackEl.textContent = "Камера включена. Вы видите себя!";
      feedbackEl.style.color = '#00ff00';
      runDetection();
    };
  } catch (err) {
    feedbackEl.textContent = "Ошибка камеры: " + err.message;
    feedbackEl.style.color = '#ff4757';
  }
});

document.getElementById('stopButton').addEventListener('click', () => {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
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
  squatStage = lungeStage = pushupStage = 'up';
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
    repCount = 0;
    repCountEl.textContent = '0';
    timerEl.textContent = '0';
    plankStartTime = 0;
  });
});

// Фото
const photoUpload = document.getElementById('photoUpload');
const photoUploadArea = document.getElementById('photoUploadArea');
const analyzeBtn = document.getElementById('analyzePhotoButton');
const clearBtn = document.getElementById('clearPhotoButton');

photoUploadArea.addEventListener('click', () => photoUpload.click());

photoUpload.addEventListener('change', () => {
  if (photoUpload.files[0]) {
    photoPreview.src = URL.createObjectURL(photoUpload.files[0]);
    photoPreview.style.display = 'block';
    analyzeBtn.disabled = false;
    clearBtn.disabled = false;
  }
});

analyzeBtn.addEventListener('click', async () => {
  await loadModel();

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
      const fb = giveFeedback(detected, landmarks);
      photoFeedbackEl.innerHTML = `<strong>${fb.message}</strong><br><small>${fb.advice}</small>`;
      photoFeedbackEl.style.color = fb.color;
    } else {
      photoFeedbackEl.textContent = 'Поза не обнаружена на фото.';
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
  canvas.width = 640;
  canvas.height = 480;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#fff';
  ctx.font = '20px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Включите камеру или загрузите фото', canvas.width / 2, canvas.height / 2);
});