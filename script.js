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
let squatStage = 'up'; // начальное положение для приседаний
let lungeStage = 'up'; // начальное положение для выпадов
let lastDetectionTime = 0;
let isVideoMode = false;
let mp = null; // для работы с Image

// Инициализация MediaPipe
async function initPoseLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  // Сохраняем mp для использования с изображениями
  mp = { Image: class {
    constructor(element, format) {
      this.image = element;
      this.format = format;
    }
  }};

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
  return Math.abs(shoulderY - hipY) < 0.08; // более строгий порог
}

function detectExercise(landmarks) {
  const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
  const rHip = landmarks[24], rKnee = landmarks[26], rAnkle = landmarks[28];
  const lShoulder = landmarks[11], rShoulder = landmarks[12];
  const nose = landmarks[0];

  // Проверка, что поза в целом видна
  const visibilityThreshold = 0.3;
  if (!nose || nose.visibility < visibilityThreshold) {
    return 'none';
  }

  const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
  const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
  const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;
  const kneeDiff = Math.abs(leftKneeAngle - rightKneeAngle);

  // Для планки проверяем угол линии тело-ноги
  const bodyLineAngle = calculateAngle(lShoulder, lHip, lAnkle);
  
  // Проверка на вертикальную стойку (стоя)
  const shoulderHipAngle = calculateAngle(lShoulder, lHip, lKnee);
  const isStandingStraight = avgKneeAngle > 160 && bodyLineAngle > 160 && shoulderHipAngle > 160;
  
  // Приседание - оба колена согнуты примерно одинаково
  if (avgKneeAngle < 140 && kneeDiff < 30) {
    return 'squats';
  }
  
  // Выпады - одно колено сильно больше согнуто, чем другое
  if (avgKneeAngle < 140 && kneeDiff > 30) {
    return 'lunges';
  }
  
  // Планка - тело горизонтально и колени почти прямые
  if (avgKneeAngle > 160 && isBodyHorizontal(landmarks) && Math.abs(bodyLineAngle - 180) < 10) {
    return 'plank';
  }
  
  // Если ничего не подошло, но человек стоит прямо
  if (isStandingStraight) {
    return 'standing'; // новая категория для стойки
  }

  return 'none';
}

function giveFeedback(exercise, landmarks) {
  feedbackEl.style.color = '#ffd93d'; // Жёлтый по умолчанию

  if (exercise === 'none' || exercise === 'standing') {
    return 'Встаньте в стартовую позицию для упражнения.';
  }

  const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
  const rHip = landmarks[24], rKnee = landmarks[26], rAnkle = landmarks[28];
  const lShoulder = landmarks[11], rShoulder = landmarks[12];
  const lElbow = landmarks[13], rElbow = landmarks[14];
  const lWrist = landmarks[15], rWrist = landmarks[16];

  let msg = '';
  const now = Date.now();

  if (exercise === 'squats') {
    const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
    const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
    const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;
    const hipAngle = calculateAngle(lShoulder, lHip, lKnee);
    const backAngle = calculateAngle(lShoulder, lHip, lKnee);

    // Советы по технике
    if (avgKneeAngle < 90) {
      msg = 'Слишком глубокий присед! Колени не должны выходить за носки слишком сильно';
      feedbackEl.style.color = '#ff4757';
    } else if (avgKneeAngle >= 90 && avgKneeAngle <= 120) {
      if (hipAngle > 140) {
        msg = 'Отлично! Идеальная глубина, спина прямая 🔥';
        feedbackEl.style.color = '#00ff00';
      } else {
        msg = 'Хорошая глубина, но держите спину прямой!';
        feedbackEl.style.color = '#ffd93d';
      }
    } else if (avgKneeAngle > 120 && avgKneeAngle < 140) {
      msg = 'Приседайте глубже (до 90-120°)';
      feedbackEl.style.color = '#ff4757';
    }

    // Логика подсчета повторений
    if (avgKneeAngle < 120 && squatStage === 'up') {
      squatStage = 'down';
      msg += ' Опускайтесь...';
    }
    
    if (avgKneeAngle > 160 && squatStage === 'down') {
      squatStage = 'up';
      repCount++;
      repCountEl.textContent = repCount;
      msg = 'Отлично! +1 повторение 💪';
      feedbackEl.style.color = '#00ff00';
    }

  } else if (exercise === 'lunges') {
    const leftAngle = calculateAngle(lHip, lKnee, lAnkle);
    const rightAngle = calculateAngle(rHip, rKnee, rAnkle);
    const frontAngle = Math.min(leftAngle, rightAngle);
    const backAngle = Math.max(leftAngle, rightAngle);

    if (frontAngle > 85 && frontAngle < 95 && backAngle > 140) {
      msg = 'Идеально! Переднее колено под 90°, заднее почти прямое 👌';
      feedbackEl.style.color = '#00ff00';
    } else if (frontAngle < 85) {
      msg = 'Согните переднюю ногу сильнее (цель 90°)';
      feedbackEl.style.color = '#ff4757';
    } else if (frontAngle > 100) {
      msg = 'Не переразгибайте переднее колено';
      feedbackEl.style.color = '#ff4757';
    } else {
      msg = 'Старайтесь, чтобы переднее колено было под 90°';
      feedbackEl.style.color = '#ffd93d';
    }

    // Логика подсчета для выпадов
    if (frontAngle < 90 && lungeStage === 'up') {
      lungeStage = 'down';
      msg += ' Опускайтесь...';
    }
    
    if (frontAngle > 140 && lungeStage === 'down') {
      lungeStage = 'up';
      repCount++;
      repCountEl.textContent = repCount;
      msg = 'Супер! +1 выпад 💪';
      feedbackEl.style.color = '#00ff00';
    }

  } else if (exercise === 'plank') {
    const lineAngle = calculateAngle(lShoulder, lHip, lAnkle);
    const shoulderHipAngle = calculateAngle(lShoulder, lHip, lKnee);
    
    if (lineAngle > 175 && Math.abs(shoulderHipAngle - 180) < 5) {
      if (plankStartTime === 0) {
        plankStartTime = Date.now();
        msg = 'Планка начата! Держите спину прямо 💪';
      } else {
        const seconds = Math.floor((Date.now() - plankStartTime) / 1000);
        timerEl.textContent = seconds;
        msg = `Держите! ${seconds} сек. Тело прямое как доска 💪`;
      }
      feedbackEl.style.color = '#00ff00';
    } else {
      if (lineAngle < 170) {
        msg = 'Провисает спина — подтяните живот и ягодицы!';
      } else if (shoulderHipAngle < 170) {
        msg = 'Таз слишком высоко — опуститесь в линию!';
      }
      feedbackEl.style.color = '#ff4757';
      plankStartTime = 0;
      timerEl.textContent = '0';
    }
  }

  return msg;
}

function processResults(results, sourceImage) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(sourceImage, 0, 0, canvas.width, canvas.height);

  if (results.landmarks && results.landmarks.length > 0) {
    const landmarks = results.landmarks[0];
    const drawingUtils = new DrawingUtils(ctx);
    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
    drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', radius: 6 });

    const detected = detectExercise(landmarks);
    
    // Обновляем текущее упражнение только если уверены в определении
    if (detected !== 'none' && detected !== 'standing') {
      currentExercise = detected;
    }

    // Сброс при смене упражнения
    if (currentExercise !== previousExercise && previousExercise !== 'none') {
      previousExercise = currentExercise;
      repCount = 0;
      repCountEl.textContent = '0';
      plankStartTime = 0;
      timerEl.textContent = '0';
      squatStage = 'up';
      lungeStage = 'up';

      const names = { 
        squats: 'Приседания', 
        lunges: 'Выпады', 
        plank: 'Планка',
        standing: 'Стойка',
        none: 'Не определено'
      };
      exerciseNameEl.textContent = names[currentExercise] || 'Определение...';
    }

    const feedbackMsg = giveFeedback(currentExercise, landmarks);
    feedbackEl.textContent = feedbackMsg;
    
  } else {
    feedbackEl.textContent = 'Поза не обнаружена. Встаньте в полный рост перед камерой.';
    feedbackEl.style.color = '#ff4757';
    exerciseNameEl.textContent = 'Не определено';
  }
}

function runVideoDetection() {
  if (!poseLandmarker || !isVideoMode) return;
  
  try {
    const results = poseLandmarker.detectForVideo(video, performance.now());
    processResults(results, video);
  } catch (error) {
    console.error('Ошибка при детекции:', error);
  }
  
  requestAnimationFrame(runVideoDetection);
}

// Инициализация при загрузке
initPoseLandmarker();

// Камера
document.getElementById('startButton').addEventListener('click', async () => {
  if (!poseLandmarker) {
    await initPoseLandmarker();
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { 
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 }
      } 
    });
    
    video.srcObject = stream;
    video.play();
    
    video.onloadedmetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      isVideoMode = true;
      
      // Сброс счетчиков при запуске видео
      repCount = 0;
      repCountEl.textContent = '0';
      plankStartTime = 0;
      timerEl.textContent = '0';
      squatStage = 'up';
      lungeStage = 'up';
      currentExercise = 'none';
      previousExercise = 'none';
      
      exerciseNameEl.textContent = 'Определение...';
      feedbackEl.textContent = 'Камера запущена. Начните упражнение!';
      feedbackEl.style.color = '#ffd93d';
      
      runVideoDetection();
    };
    
  } catch (err) {
    feedbackEl.textContent = "Ошибка камеры: " + err.message;
    feedbackEl.style.color = '#ff4757';
  }
});

// Фото
document.getElementById('analyzePhotoButton').addEventListener('click', async () => {
  const fileInput = document.getElementById('photoUpload');
  
  if (!fileInput.files || fileInput.files.length === 0) {
    feedbackEl.textContent = 'Сначала выберите фото!';
    feedbackEl.style.color = '#ff4757';
    return;
  }

  if (!poseLandmarker) {
    await initPoseLandmarker();
  }

  isVideoMode = false; // отключаем видео режим
  
  const file = fileInput.files[0];
  const img = new Image();
  
  img.onload = async () => {
    // Устанавливаем размеры canvas под фото
    const maxWidth = 800;
    const maxHeight = 600;
    let width = img.width;
    let height = img.height;
    
    if (width > maxWidth) {
      height = (maxWidth / width) * height;
      width = maxWidth;
    }
    if (height > maxHeight) {
      width = (maxHeight / height) * width;
      height = maxHeight;
    }
    
    canvas.width = width;
    canvas.height = height;
    
    try {
      // Создаем временный canvas для рисования и получения данных
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = img.width;
      tempCanvas.height = img.height;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(img, 0, 0);
      
      // Создаем ImageData для MediaPipe
      const imageData = tempCtx.getImageData(0, 0, img.width, img.height);
      
      // Для фото используем IMAGE режим
      const imageModePoseLandmarker = await PoseLandmarker.createFromOptions(
        await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"),
        {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU"
          },
          runningMode: "IMAGE",
          numPoses: 1
        }
      );
      
      const results = imageModePoseLandmarker.detect(imageData);
      
      // Рисуем изображение на основном canvas
      ctx.drawImage(img, 0, 0, width, height);
      
      if (results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];
        const drawingUtils = new DrawingUtils(ctx);
        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
        drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', radius: 6 });

        const detected = detectExercise(landmarks);
        const names = { 
          squats: 'Приседания', 
          lunges: 'Выпады', 
          plank: 'Планка',
          standing: 'Стойка',
          none: 'Не определено'
        };
        
        exerciseNameEl.textContent = names[detected] || 'Определение...';
        
        const feedbackMsg = giveFeedback(detected, landmarks);
        feedbackEl.textContent = feedbackMsg || 'Поза определена. ' + names[detected];
        
      } else {
        feedbackEl.textContent = 'Не удалось определить позу на фото. Убедитесь, что человек виден полностью.';
        feedbackEl.style.color = '#ff4757';
        exerciseNameEl.textContent = 'Не определено';
      }
      
    } catch (error) {
      console.error('Ошибка при анализе фото:', error);
      feedbackEl.textContent = 'Ошибка при анализе фото: ' + error.message;
      feedbackEl.style.color = '#ff4757';
    }
  };
  
  img.onerror = () => {
    feedbackEl.textContent = 'Ошибка загрузки фото';
    feedbackEl.style.color = '#ff4757';
  };
  
  img.src = URL.createObjectURL(file);
});

// Загрузка примера фото
document.getElementById('loadExampleButton').addEventListener('click', () => {
  // Можно добавить примеры фото для тестирования
  feedbackEl.textContent = 'Загрузите свое фото для анализа';
  feedbackEl.style.color = '#ffd93d';
});