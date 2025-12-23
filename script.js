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
let plankSeconds = 0;
let plankStartTime = 0;
let currentExercise = 'none'; // 'squats', 'lunges', 'plank', 'none'
let previousExercise = 'none';
let squatStage = null; // 'up', 'down'
let lungeStage = null;
let lastAngles = []; // Для сглаживания
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
  // Ключевые точки (левая и правая сторона)
  const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
  const rHip = landmarks[24], rKnee = landmarks[26], rAnkle = landmarks[28];
  const lShoulder = landmarks[11], rShoulder = landmarks[12];
  const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
  const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
  const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;
  const leftHipAngle = calculateAngle(lShoulder, lHip, lKnee);
  const rightHipAngle = calculateAngle(rShoulder, rHip, rKnee);
  const bodyLineAngle = calculateAngle(lShoulder, lHip, lAnkle); // Для планки
  // Асимметрия ног (для выпадов)
  const kneeDiff = Math.abs(leftKneeAngle - rightKneeAngle);
  // Определяем упражнение
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
  let color = '#ff4757'; // Красный по умолчанию (неправильно)
  if (exercise === 'squats') {
    const kneeAngle = calculateAngle(lHip, lKnee, lAnkle);
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
    // Точный счёт reps
    if (kneeAngle < 95) squatStage = 'down';
    if (kneeAngle > 155 && squatStage === 'down') {
      squatStage = 'up';
      repCount++;
      repCountEl.textContent = repCount;
      msg = 'Отличное повторение! +1 💪';
      color = '#00ff00';
    }
  } else if (exercise === 'lunges') {
    // Определяем переднюю ногу (та, что сильнее согнута)
    const leftAngle = calculateAngle(lHip, lKnee, lAnkle);
    const rightAngle = calculateAngle(rHip, rKnee, rAnkle);
    const frontAngle = leftAngle < rightAngle ? leftAngle : rightAngle;
    if (frontAngle > 80 && frontAngle < 100) {
      msg = 'Идеально! Переднее колено под 90° 👌';
      color = '#00ff00';
    } else if (frontAngle < 80) {
      msg = 'Согните переднюю ногу сильнее';
    } else {
      msg = 'Не переразгибайте переднее колено';
    }
    // Счёт reps
    if (frontAngle < 85) lungeStage = 'down';
    if (frontAngle > 140 && lungeStage === 'down') {
      lungeStage = 'up';
      repCount++;
      repCountEl.textContent = repCount;
    }
  } else if (exercise === 'plank') {
    const lineAngle = calculateAngle(landmarks[11], landmarks[23], landmarks[27]);
    if (lineAngle > 170) {
      if (plankStartTime === 0) plankStartTime = Date.now();
      plankSeconds = Math.floor((Date.now() - plankStartTime) / 1000);
      timerEl.textContent = plankSeconds;
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
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (results.landmarks && results.landmarks.length > 0) {
    const landmarks = results.landmarks[0];
    const drawingUtils = new DrawingUtils(ctx);
    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
    drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', radius: 6 });
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
  }
  ctx.restore();
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