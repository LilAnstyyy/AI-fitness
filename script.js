import { FilesetResolver, PoseLandmarker, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById('webcam');
const canvas = document.getElementById('output_canvas');
const ctx = canvas.getContext('2d');
const exerciseSelect = document.getElementById('exerciseSelect');
const repCountEl = document.getElementById('repCount');
const timerEl = document.getElementById('timer');
const feedbackEl = document.getElementById('feedback');
const exerciseNameEl = document.getElementById('exerciseName');

let poseLandmarker = null;
let repCount = 0;
let stage = null; // 'up' или 'down'
let plankStartTime = 0;
let currentExercise = 'squats';

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

  feedbackEl.textContent = "Камера готова! Начните упражнение.";
}

function calculateAngle(a, b, c) {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs(radians * 180.0 / Math.PI);
  if (angle > 180) angle = 360 - angle;
  return angle;
}

function runDetection() {
  if (!poseLandmarker) return;

  const results = poseLandmarker.detectForVideo(video, performance.now());

  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (results.landmarks && results.landmarks.length > 0) {
    const landmarks = results.landmarks[0];
    const drawingUtils = new DrawingUtils(ctx);

    // Рисуем скелет
    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
    drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', radius: 6 });

    // Логика по упражнению
    if (currentExercise === 'squats') {
      const hip = landmarks[23], knee = landmarks[25], ankle = landmarks[27];
      const angle = calculateAngle(hip, knee, ankle);
      feedbackEl.textContent = `Угол колена: ${Math.round(angle)}°`;

      if (angle < 90) stage = 'down';
      if (angle > 160 && stage === 'down') {
        stage = 'up';
        repCount++;
        repCountEl.textContent = repCount;
        feedbackEl.textContent = 'Отличное повторение! 🔥';
      }

    } else if (currentExercise === 'lunges') {
      // Передняя левая нога (для болгарских выпадов)
      const hip = landmarks[23], knee = landmarks[25], ankle = landmarks[27];
      const angle = calculateAngle(hip, knee, ankle);

      if (angle > 80 && angle < 100) {
        feedbackEl.textContent = 'Идеально! Переднее колено под 90° 👌';
      } else if (angle < 80) {
        feedbackEl.textContent = 'Согните переднюю ногу сильнее';
      } else {
        feedbackEl.textContent = 'Не переразгибайте колено';
      }

      if (angle < 85) stage = 'down';
      if (angle > 150 && stage === 'down') {
        stage = 'up';
        repCount++;
        repCountEl.textContent = repCount;
      }

    } else if (currentExercise === 'plank') {
      const shoulder = landmarks[11], hip = landmarks[23], ankle = landmarks[27];
      const angle = calculateAngle(shoulder, hip, ankle);

      if (angle > 170) {
        if (plankStartTime === 0) plankStartTime = Date.now();
        const seconds = Math.floor((Date.now() - plankStartTime) / 1000);
        timerEl.textContent = seconds;
        feedbackEl.textContent = 'Держите прямую линию! 💪';
      } else {
        feedbackEl.textContent = 'Выровняйте тело — спина провисает!';
        if (plankStartTime !== 0) plankStartTime = 0;
        timerEl.textContent = '0';
      }
    }
  }

  ctx.restore();
  requestAnimationFrame(runDetection);
}

document.getElementById('startButton').addEventListener('click', async () => {
  if (!poseLandmarker) {
    await initPoseLandmarker();
  }

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
      feedbackEl.textContent = "Ошибка доступа к камере: " + err.message;
      console.error(err);
    });
});

exerciseSelect.addEventListener('change', (e) => {
  currentExercise = e.target.value;
  exerciseNameEl.textContent = e.target.options[e.target.selectedIndex].text;
  repCount = 0;
  repCountEl.textContent = '0';
  timerEl.textContent = '0';
  plankStartTime = 0;
  stage = null;
  feedbackEl.textContent = 'Начните упражнение!';
});