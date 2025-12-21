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
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  feedbackEl.textContent = "Модель готова. Встаньте в кадр полностью!";
}

function calculateAngle(a, b, c) {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs(radians * 180.0 / Math.PI);
  if (angle > 180) angle = 360 - angle;
  return angle;
}

function isBodyHorizontal(landmarks) {
  const lShoulder = landmarks[11];
  const lHip = landmarks[23];
  const lKnee = landmarks[25];
  const hipToShoulderY = Math.abs(lHip.y - lShoulder.y);
  const hipToKneeY = Math.abs(lHip.y - lKnee.y);
  return hipToShoulderY < 0.2 && hipToKneeY > 0.3; // Тело горизонтально, ноги вниз
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

  // Сначала проверяем squats или lunges (если колени согнуты)
  if (avgKneeAngle < 150) {
    if (kneeDiff > 35) return 'lunges';
    return 'squats';
  }

  // Только если колени прямые и тело горизонтальное — plank
  if (avgKneeAngle > 160 && bodyLineAngle > 160 && isBodyHorizontal(landmarks)) {
    return 'plank';
  }

  return 'none'; // Стоячее положение или неясно
}

// giveFeedback и processResults остаются как в предыдущей версии (с фиксом перерисовки видео)

function processResults(results, isVideo = true) {
  if (isVideo) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  }

  if (results.landmarks && results.landmarks.length > 0) {
    const landmarks = results.landmarks[0];
    const drawingUtils = new DrawingUtils(ctx);
    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
    drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', radius: 6 });

    const detected = detectExercise(landmarks);
    if (detected !== 'none') currentExercise = detected;

    if (currentExercise !== previousExercise) {
      previousExercise = currentExercise;
      repCount = 0; repCountEl.textContent = '0';
      plankStartTime = 0; timerEl.textContent = '0';
      squatStage = null; lungeStage = null;

      const names = { squats: 'Приседания', lunges: 'Выпады (болгарские)', plank: 'Планка' };
      exerciseNameEl.textContent = names[currentExercise] || 'Стартовая позиция';
    }

    feedbackEl.textContent = giveFeedback(currentExercise, landmarks);
  } else {
    if (isVideo) ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    feedbackEl.textContent = 'Поза не обнаружена. Попробуйте фронтальный или полу-боковой ракурс.';
    feedbackEl.style.color = '#ffd93d';
  }
}

// runVideoDetection и обработчики камеры/фото — без изменений (с фиксом ctx.drawImage)
