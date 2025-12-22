=import { FilesetResolver, PoseLandmarker, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

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

// Создаем кнопку для переключения скелета, если её нет в HTML
function createToggleSkeletonButton() {
  if (!document.getElementById('toggleSkeleton')) {
    const controlsDiv = document.querySelector('.controls');
    if (controlsDiv) {
      const toggleBtn = document.createElement('button');
      toggleBtn.id = 'toggleSkeleton';
      toggleBtn.className = 'btn btn-secondary';
      toggleBtn.textContent = 'Скрыть скелет';
      toggleBtn.disabled = true;
      controlsDiv.appendChild(toggleBtn);
    }
  }
}

function getDetailedAdvice(exercise, landmarks) {
  const advice = [];
  
  if (exercise === 'squats') {
    const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
    const rHip = landmarks[24], rKnee = landmarks[26], rAnkle = landmarks[28];
    const lShoulder = landmarks[11];
    
    const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
    const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
    const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;
    const hipAngle = calculateAngle(lShoulder, lHip, lKnee);
    
    if (avgKneeAngle > 120) {
      advice.push("• Приседайте глубже (угол в коленях должен быть около 90°)");
    }
    if (hipAngle < 140) {
      advice.push("• Держите спину прямой, грудь вперед");
    }
    if (Math.abs(leftKneeAngle - rightKneeAngle) > 20) {
      advice.push("• Равномерно распределяйте вес на обе ноги");
    }
    if (advice.length === 0) {
      advice.push("• Отличная техника! Продолжайте в том же духе");
    }
    
  } else if (exercise === 'lunges') {
    const leftAngle = calculateAngle(landmarks[23], landmarks[25], landmarks[27]);
    const rightAngle = calculateAngle(landmarks[24], landmarks[26], landmarks[28]);
    const frontAngle = Math.min(leftAngle, rightAngle);
    const backAngle = Math.max(leftAngle, rightAngle);
    
    if (frontAngle > 100) {
      advice.push("• Согните переднее колено сильнее (цель 90°)");
    } else if (frontAngle < 80) {
      advice.push("• Не опускайтесь слишком низко, переднее колено должно быть под 90°");
    }
    if (backAngle < 120) {
      advice.push("• Задняя нога должна быть почти прямой");
    }
    if (Math.abs(landmarks[23].y - landmarks[24].y) > 0.1) {
      advice.push("• Держите таз ровно, не заваливайтесь в сторону");
    }
    if (advice.length === 0) {
      advice.push("• Идеальный выпад! Колено не выходит за носок");
    }
    
  } else if (exercise === 'plank') {
    const lineAngle = calculateAngle(landmarks[11], landmarks[23], landmarks[27]);
    const shoulderHipAngle = calculateAngle(landmarks[11], landmarks[23], landmarks[25]);
    
    if (lineAngle < 170) {
      advice.push("• Подтяните живот и ягодицы, чтобы тело было прямым");
    }
    if (shoulderHipAngle < 170) {
      advice.push("• Опустите таз, чтобы тело образовало прямую линию");
    }
    if (landmarks[23].y < landmarks[11].y) {
      advice.push("• Таз слишком высоко, опустите его");
    }
    if (advice.length === 0) {
      advice.push("• Отличная планка! Тело прямое как струна");
    }
    
  } else if (exercise === 'pushups') {
    const lElbow = landmarks[13], rElbow = landmarks[14];
    const lWrist = landmarks[15], rWrist = landmarks[16];
    const leftElbowAngle = calculateAngle(landmarks[11], lElbow, lWrist);
    const rightElbowAngle = calculateAngle(landmarks[12], rElbow, rWrist);
    const avgElbowAngle = (leftElbowAngle + rightElbowAngle) / 2;
    
    if (avgElbowAngle > 130) {
      advice.push("• Опуститесь глубже, локти должны сгибаться до 90°");
    }
    if (Math.abs(leftElbowAngle - rightElbowAngle) > 20) {
      advice.push("• Держите локти симметрично");
    }
    const bodyLineAngle = calculateAngle(landmarks[11], landmarks[23], landmarks[27]);
    if (bodyLineAngle < 170) {
      advice.push("• Держите тело прямой линией, не прогибайтесь в пояснице");
    }
    if (advice.length === 0) {
      advice.push("• Отличная техника отжиманий!");
    }
  }
  
  return advice.join('\n');
}

async function loadModel() {
  if (poseLandmarker) return;

  feedbackEl.textContent = "Загрузка модели ИИ...";
  photoFeedbackEl.textContent = "Загрузка модели ИИ...";

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
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  feedbackEl.textContent = "Модель загружена! Включите камеру или загрузите фото.";
  photoFeedbackEl.textContent = "Модель загружена. Загрузите фото для анализа.";
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
  return Math.abs(shoulderY - hipY) < 0.1;
}

function detectExercise(landmarks) {
  // Проверка видимости ключевых точек
  const keyPoints = [11, 12, 23, 24, 25, 26];
  const avgVisibility = keyPoints.reduce((sum, i) => sum + (landmarks[i]?.visibility || 0), 0) / keyPoints.length;
  if (avgVisibility < 0.3) return 'none';

  const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
  const rHip = landmarks[24], rKnee = landmarks[26], rAnkle = landmarks[28];
  const lShoulder = landmarks[11], rShoulder = landmarks[12];
  const lElbow = landmarks[13], rElbow = landmarks[14];
  const lWrist = landmarks[15], rWrist = landmarks[16];

  const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
  const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
  const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;
  const kneeDiff = Math.abs(leftKneeAngle - rightKneeAngle);
  const bodyLineAngle = calculateAngle(lShoulder, lHip, lAnkle);
  
  // Проверка на отжимания
  const leftElbowAngle = calculateAngle(lShoulder, lElbow, lWrist);
  const rightElbowAngle = calculateAngle(rShoulder, rElbow, rWrist);
  const avgElbowAngle = (leftElbowAngle + rightElbowAngle) / 2;
  const isPushupPosition = avgElbowAngle < 150 && isBodyHorizontal(landmarks);

  if (isPushupPosition && avgElbowAngle < 120) {
    return 'pushups';
  } else if (avgKneeAngle < 140) {
    if (kneeDiff > 30) return 'lunges';
    return 'squats';
  } else if (isBodyHorizontal(landmarks) && bodyLineAngle > 175 && avgKneeAngle > 160) {
    return 'plank';
  }

  return 'none'; // Вертикальная стойка или неопределено
}

function giveFeedback(exercise, landmarks) {
  if (exercise === 'none') {
    return {
      message: 'Стойка. Встаньте в позицию для упражнения.',
      color: '#ffcc00',
      advice: 'Для лучшего определения:\n• Встаньте боком к камере\n• Убедитесь, что все тело в кадре\n• Носите обтягивающую одежду'
    };
  }

  let message = '';
  let color = '#ff4757';
  let advice = '';

  if (exercise === 'squats') {
    const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
    const rHip = landmarks[24], rKnee = landmarks[26], rAnkle = landmarks[28];
    const lShoulder = landmarks[11];
    
    const avgKneeAngle = (calculateAngle(lHip, lKnee, lAnkle) + calculateAngle(rHip, rKnee, rAnkle)) / 2;
    const hipAngle = calculateAngle(lShoulder, lHip, lKnee);

    if (avgKneeAngle < 100 && hipAngle > 140) {
      message = 'Отлично! Глубокий присед, спина прямая 🔥';
      color = '#00ff00';
    } else if (avgKneeAngle < 100) {
      message = 'Глубоко, но спина наклоняется';
    } else if (avgKneeAngle < 120) {
      message = 'Хорошо, можно присесть глубже';
      color = '#ffcc00';
    } else {
      message = 'Начните приседание';
    }

    if (avgKneeAngle < 95 && squatStage === 'up') {
      squatStage = 'down';
      message = 'Опускаемся...';
      color = '#ffcc00';
    }
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

    advice = getDetailedAdvice('squats', landmarks);

  } else if (exercise === 'lunges') {
    const leftAngle = calculateAngle(landmarks[23], landmarks[25], landmarks[27]);
    const rightAngle = calculateAngle(landmarks[24], landmarks[26], landmarks[28]);
    const frontAngle = Math.min(leftAngle, rightAngle);

    if (frontAngle > 85 && frontAngle < 95) {
      message = 'Идеально! Переднее колено под 90° 👌';
      color = '#00ff00';
    } else if (frontAngle < 85) {
      message = 'Согните переднюю ногу сильнее';
    } else {
      message = 'Не переразгибайте переднее колено';
    }

    if (frontAngle < 90 && lungeStage === 'up') {
      lungeStage = 'down';
      message = 'Опускаемся в выпад...';
      color = '#ffcc00';
    }
    if (frontAngle > 140 && lungeStage === 'down') {
      if (Date.now() - lastRepTime > minRepInterval) {
        lungeStage = 'up';
        repCount++;
        repCountEl.textContent = repCount;
        lastRepTime = Date.now();
        message = 'Отлично! +1 выпад 💪';
        color = '#00ff00';
      }
    }

    advice = getDetailedAdvice('lunges', landmarks);

  } else if (exercise === 'plank') {
    const lineAngle = calculateAngle(landmarks[11], landmarks[23], landmarks[27]);
    if (lineAngle > 175) {
      if (plankStartTime === 0) {
        plankStartTime = Date.now();
        message = 'Планка начата! Держитесь 💪';
        color = '#ffcc00';
      } else {
        const seconds = Math.floor((Date.now() - plankStartTime) / 1000);
        timerEl.textContent = seconds;
        message = `Держите! ${seconds} сек. Тело прямое 🔥`;
        if (seconds > 30) message += ' Отлично!';
        color = '#00ff00';
      }
    } else {
      message = 'Провисает спина или таз — выпрямитесь!';
      plankStartTime = 0;
      timerEl.textContent = '0';
    }
    
    advice = getDetailedAdvice('plank', landmarks);
    
  } else if (exercise === 'pushups') {
    const lElbow = landmarks[13], rElbow = landmarks[14];
    const lWrist = landmarks[15], rWrist = landmarks[16];
    const avgElbowAngle = (calculateAngle(landmarks[11], lElbow, lWrist) + 
                          calculateAngle(landmarks[12], rElbow, rWrist)) / 2;
    
    if (avgElbowAngle < 100) {
      message = 'Отлично! Полная амплитуда 💪';
      color = '#00ff00';
    } else if (avgElbowAngle < 130) {
      message = 'Хорошо, опуститесь еще немного';
      color = '#ffcc00';
    } else {
      message = 'Начните отжимание';
    }
    
    // Подсчет повторений для отжиманий
    if (avgElbowAngle < 100 && pushupStage === 'up') {
      pushupStage = 'down';
      message = 'Опускаемся...';
      color = '#ffcc00';
    }
    if (avgElbowAngle > 150 && pushupStage === 'down') {
      if (Date.now() - lastRepTime > minRepInterval) {
        pushupStage = 'up';
        repCount++;
        repCountEl.textContent = repCount;
        lastRepTime = Date.now();
        message = 'Отлично! +1 отжимание 💪';
        color = '#00ff00';
      }
    }
    
    advice = getDetailedAdvice('pushups', landmarks);
  }

  return { message, color, advice };
}

function processVideoFrame(results) {
  // Очищаем canvas и рисуем видео
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (results.landmarks && results.landmarks.length > 0) {
    const landmarks = results.landmarks[0];
    
    // Рисуем скелет поверх видео
    if (isDrawingEnabled) {
      const drawingUtils = new DrawingUtils(ctx);
      drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { 
        color: '#00FF00', 
        lineWidth: 3
      });
      drawingUtils.drawLandmarks(landmarks, { 
        color: '#FF0000', 
        radius: 4
      });
    }

    let detected = currentExercise === 'auto' ? detectExercise(landmarks) : currentExercise;
    if (detected === 'none' && currentExercise !== 'auto') detected = currentExercise;

    exerciseNameEl.textContent = EXERCISE_NAMES[detected] || 'Определение...';
    const feedback = giveFeedback(detected, landmarks);
    
    // Обновляем текст обратной связи
    if (feedback.advice) {
      feedbackEl.innerHTML = `<strong>${feedback.message}</strong><br><small style="color: #aaa; white-space: pre-line;">${feedback.advice}</small>`;
    } else {
      feedbackEl.textContent = feedback.message;
    }
    feedbackEl.style.color = feedback.color;
    
  } else {
    feedbackEl.textContent = 'Поза не обнаружена. Убедитесь, что:\n• Все тело в кадре\n• Хорошее освещение\n• Вы стоите боком к камере';
    feedbackEl.style.color = '#ff4757';
    exerciseNameEl.textContent = '—';
  }
}

function runVideoDetection() {
  if (!isCameraRunning || !poseLandmarker) return;
  try {
    const results = poseLandmarker.detectForVideo(video, performance.now());
    processVideoFrame(results);
    requestAnimationFrame(runVideoDetection);
  } catch (error) {
    console.error('Detection error:', error);
    feedbackEl.textContent = 'Ошибка определения. Перезапустите камеру.';
    feedbackEl.style.color = '#ff4757';
  }
}

// Камера
document.getElementById('startButton').addEventListener('click', async () => {
  if (isCameraRunning) return;

  await loadModel();
  createToggleSkeletonButton(); // Создаем кнопку если её нет

  try {
    // Запрашиваем камеру
    stream = await navigator.mediaDevices.getUserMedia({
      video: { 
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });
    
    video.srcObject = stream;
    
    // Ждем загрузки метаданных видео
    await new Promise((resolve) => {
      video.onloadedmetadata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        resolve();
      };
    });
    
    await video.play();
    
    isCameraRunning = true;
    document.getElementById('startButton').disabled = true;
    document.getElementById('stopButton').disabled = false;
    
    // Активируем кнопку переключения скелета если она есть
    const toggleSkeletonBtn = document.getElementById('toggleSkeleton');
    if (toggleSkeletonBtn) {
      toggleSkeletonBtn.disabled = false;
    }
    
    // Сбрасываем счетчики при запуске камеры
    repCount = 0;
    plankStartTime = 0;
    squatStage = 'up';
    lungeStage = 'up';
    pushupStage = 'up';
    lastRepTime = 0;
    repCountEl.textContent = '0';
    timerEl.textContent = '0';
    
    feedbackEl.textContent = "Камера включена! Вы видите себя. Начните упражнение.";
    feedbackEl.style.color = '#00ff00';
    
    runVideoDetection();
    
  } catch (err) {
    console.error('Camera error:', err);
    feedbackEl.textContent = "Ошибка камеры: " + (err.message || "Не удалось получить доступ к камере");
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
  
  // Деактивируем кнопку переключения скелета если она есть
  const toggleSkeletonBtn = document.getElementById('toggleSkeleton');
  if (toggleSkeletonBtn) {
    toggleSkeletonBtn.disabled = true;
  }
  
  // Очищаем canvas и показываем сообщение
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#333';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#fff';
  ctx.font = '20px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Камера выключена', canvas.width/2, canvas.height/2 - 20);
  ctx.fillText('Нажмите "Включить камеру"', canvas.width/2, canvas.height/2 + 20);
  
  feedbackEl.textContent = "Камера выключена. Для продолжения включите камеру.";
  feedbackEl.style.color = '#ffcc00';
});

document.getElementById('resetButton').addEventListener('click', () => {
  repCount = 0;
  plankStartTime = 0;
  squatStage = 'up';
  lungeStage = 'up';
  pushupStage = 'up';
  lastRepTime = 0;
  repCountEl.textContent = '0';
  timerEl.textContent = '0';
  feedbackEl.textContent = "Счетчики сброшены. Начинаем заново!";
  feedbackEl.style.color = '#ffcc00';
});

// Кнопка включения/выключения скелета (обработчик добавляется динамически)
document.addEventListener('DOMContentLoaded', function() {
  const toggleSkeletonBtn = document.getElementById('toggleSkeleton');
  if (!toggleSkeletonBtn) {
    // Создаем кнопку при загрузке страницы
    createToggleSkeletonButton();
  }
  
  // Добавляем обработчик для кнопки переключения скелета
  document.addEventListener('click', function(e) {
    if (e.target && e.target.id === 'toggleSkeleton') {
      isDrawingEnabled = !isDrawingEnabled;
      e.target.textContent = isDrawingEnabled ? 'Скрыть скелет' : 'Показать скелет';
      e.target.classList.toggle('btn-secondary');
      e.target.classList.toggle('btn-primary');
    }
  });
});

// Выбор упражнения
document.querySelectorAll('.exercise-btn').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.exercise-btn').forEach(b => b.classList.remove('active'));
    this.classList.add('active');
    currentExercise = this.dataset.exercise;
    
    // Сбрасываем счетчики при смене упражнения
    repCount = 0;
    plankStartTime = 0;
    squatStage = 'up';
    lungeStage = 'up';
    pushupStage = 'up';
    lastRepTime = 0;
    repCountEl.textContent = '0';
    timerEl.textContent = '0';
    
    const exerciseName = currentExercise === 'auto' ? 'Автоопределение' : EXERCISE_NAMES[currentExercise];
    feedbackEl.textContent = `Упражнение: ${exerciseName}. Начинайте!`;
    feedbackEl.style.color = '#ffcc00';
  });
});

// Обработка фото
const photoUpload = document.getElementById('photoUpload');
const photoUploadArea = document.getElementById('photoUploadArea');
const analyzeBtn = document.getElementById('analyzePhotoButton');
const clearBtn = document.getElementById('clearPhotoButton');

photoUploadArea.addEventListener('click', () => photoUpload.click());

// Drag and drop для фото
photoUploadArea.addEventListener('dragover', (e) => {
  e.preventDefault();
  photoUploadArea.style.backgroundColor = 'rgba(0, 255, 0, 0.1)';
});

photoUploadArea.addEventListener('dragleave', () => {
  photoUploadArea.style.backgroundColor = '';
});

photoUploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  photoUploadArea.style.backgroundColor = '';
  
  if (e.dataTransfer.files.length) {
    photoUpload.files = e.dataTransfer.files;
    handlePhotoSelection();
  }
});

photoUpload.addEventListener('change', handlePhotoSelection);

function handlePhotoSelection() {
  if (photoUpload.files && photoUpload.files[0]) {
    const file = photoUpload.files[0];
    
    // Проверка типа файла
    if (!file.type.match('image.*')) {
      photoFeedbackEl.textContent = "Пожалуйста, выберите изображение (JPG, PNG)";
      photoFeedbackEl.style.color = '#ff4757';
      return;
    }
    
    // Проверка размера файла
    if (file.size > 5 * 1024 * 1024) {
      photoFeedbackEl.textContent = "Файл слишком большой (максимум 5MB)";
      photoFeedbackEl.style.color = '#ff4757';
      return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
      photoPreview.src = e.target.result;
      photoPreview.style.display = 'block';
      analyzeBtn.disabled = false;
      clearBtn.disabled = false;
      photoFeedbackEl.textContent = "Фото загружено. Нажмите 'Анализировать'";
      photoFeedbackEl.style.color = '#ffcc00';
    };
    reader.readAsDataURL(file);
  }
}

analyzeBtn.addEventListener('click', async () => {
  await loadModel();
  
  if (!photoPreview.src) {
    photoFeedbackEl.textContent = "Сначала загрузите фото";
    photoFeedbackEl.style.color = '#ff4757';
    return;
  }
  
  analyzeBtn.disabled = true;
  const originalText = analyzeBtn.textContent;
  analyzeBtn.textContent = 'Анализ...';
  photoFeedbackEl.textContent = "Анализ позы...";
  photoFeedbackEl.style.color = '#ffcc00';
  
  const img = new Image();
  img.src = photoPreview.src;
  
  img.onload = async () => {
    try {
      // Создаем временный canvas для анализа
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = img.width;
      tempCanvas.height = img.height;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.drawImage(img, 0, 0);
      
      // Получаем ImageData для анализа
      const imageData = tempCtx.getImageData(0, 0, img.width, img.height);
      
      // Анализируем фото
      const results = poseLandmarker.detect(imageData);
      
      // Очищаем основной canvas и рисуем фото
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      
      if (results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];
        
        // Рисуем скелет поверх фото
        const drawingUtils = new DrawingUtils(ctx);
        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { 
          color: '#00FF00', 
          lineWidth: 3 
        });
        drawingUtils.drawLandmarks(landmarks, { 
          color: '#FF0000', 
          radius: 4 
        });
        
        const detected = currentExercise === 'auto' ? detectExercise(landmarks) : currentExercise;
        const exerciseName = EXERCISE_NAMES[detected] || 'Стойка';
        photoExerciseNameEl.textContent = exerciseName;
        
        const feedback = giveFeedback(detected, landmarks);
        photoFeedbackEl.innerHTML = `<strong>${feedback.message}</strong><br><small style="white-space: pre-line;">${feedback.advice || ''}</small>`;
        photoFeedbackEl.style.color = feedback.color;
        
      } else {
        photoExerciseNameEl.textContent = '—';
        photoFeedbackEl.textContent = 'Поза не обнаружена. Попробуйте:\n• Фото с полным видом тела\n• Боковой ракурс\n• Хорошее освещение';
        photoFeedbackEl.style.color = '#ff4757';
      }
      
    } catch (error) {
      console.error('Photo analysis error:', error);
      photoFeedbackEl.textContent = 'Ошибка анализа. Попробуйте другое фото.';
      photoFeedbackEl.style.color = '#ff4757';
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = originalText;
    }
  };
  
  img.onerror = () => {
    photoFeedbackEl.textContent = 'Ошибка загрузки фото';
    photoFeedbackEl.style.color = '#ff4757';
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = originalText;
  };
});

clearBtn.addEventListener('click', () => {
  photoPreview.src = '';
  photoPreview.style.display = 'none';
  photoUpload.value = '';
  photoExerciseNameEl.textContent = '—';
  photoFeedbackEl.textContent = 'Загрузите фото для анализа';
  photoFeedbackEl.style.color = '#aaa';
  analyzeBtn.disabled = true;
  clearBtn.disabled = true;
  
  // Очищаем canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Если камера выключена, показываем сообщение
  if (!isCameraRunning) {
    ctx.fillStyle = '#333';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#fff';
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Загрузите фото для анализа', canvas.width/2, canvas.height/2 - 20);
    ctx.fillText('или включите камеру', canvas.width/2, canvas.height/2 + 20);
  }
});

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
  // Устанавливаем размеры canvas по умолчанию
  canvas.width = 640;
  canvas.height = 480;
  
  // Отрисовываем начальный экран
  ctx.fillStyle = '#333';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#fff';
  ctx.font = '20px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Включите камеру для начала', canvas.width/2, canvas.height/2 - 20);
  ctx.fillText('или загрузите фото для анализа', canvas.width/2, canvas.height/2 + 20);
  
  feedbackEl.textContent = "Добро пожаловать! Включите камеру или загрузите фото для анализа упражнений.";
  feedbackEl.style.color = '#ffcc00';
  
  // Создаем кнопку переключения скелета
  createToggleSkeletonButton();
});