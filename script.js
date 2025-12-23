import { FilesetResolver, PoseLandmarker, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById('webcam');
const canvas = document.getElementById('output_canvas');
const ctx = canvas.getContext('2d');
const repCountEl = document.getElementById('repCount');
const timerEl = document.getElementById('timer');
const feedbackEl = document.getElementById('feedback');
const exerciseNameEl = document.getElementById('exerciseName');

let poseLandmarkerVideo = null;
let poseLandmarkerImage = null;
let currentPoseLandmarker = null;

// Счётчики и состояния
let repCount = 0;
let plankStartTime = 0;
let currentExercise = 'none';
let previousExercise = 'none';
let squatStage = 'up';
let lungeStage = 'standing';

// Стабилизация
const HISTORY_LENGTH = 3; // Уменьшено для более быстрого отклика
let exerciseHistory = new Array(HISTORY_LENGTH).fill('none');
let historyIndex = 0;

// Таймер для сброса
let lastKnownExerciseTime = 0;
const RESET_AFTER_NONE_MS = 2000;

// Флаг для отслеживания режима
let isPhotoMode = false;

/**
 * Инициализация моделей
 */
async function initPoseLandmarkerVideo() {
    if (poseLandmarkerVideo) return poseLandmarkerVideo;
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    poseLandmarkerVideo = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1
    });
    return poseLandmarkerVideo;
}

async function initPoseLandmarkerImage() {
    if (poseLandmarkerImage) return poseLandmarkerImage;
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    poseLandmarkerImage = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
            delegate: "GPU"
        },
        runningMode: "IMAGE",
        numPoses: 1
    });
    return poseLandmarkerImage;
}

/**
 * Вычисление угла между тремя точками (A-B-C)
 */
function calculateAngle(a, b, c) {
    const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
    let angle = Math.abs(radians * 180.0 / Math.PI);
    if (angle > 180) angle = 360 - angle;
    return angle;
}

/**
 * Находим самое частое значение в массиве
 */
function mostFrequent(arr) {
    const count = {};
    arr.forEach(x => { 
        if (x !== 'none') {
            count[x] = (count[x] || 0) + 1; 
        }
    });
    const entries = Object.entries(count);
    if (entries.length === 0) return 'none';
    return entries.reduce((a, b) => a[1] > b[1] ? a[0] : b[0], 'none');
}

/**
 * УЛУЧШЕННАЯ и точная функция определения упражнения
 */
function detectRawExercise(landmarks) {
    if (!landmarks || landmarks.length < 29) {
        return 'none';
    }

    const nose = landmarks[0];
    const lShoulder = landmarks[11], rShoulder = landmarks[12];
    const lElbow = landmarks[13], rElbow = landmarks[14];
    const lWrist = landmarks[15], rWrist = landmarks[16];
    const lHip = landmarks[23], rHip = landmarks[24];
    const lKnee = landmarks[25], rKnee = landmarks[26];
    const lAnkle = landmarks[27], rAnkle = landmarks[28];
    const lHeel = landmarks[29], rHeel = landmarks[30];

    // Вычисляем ключевые углы
    const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
    const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
    const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;
    
    // Углы в локтях (для планки)
    const leftElbowAngle = calculateAngle(lShoulder, lElbow, lWrist);
    const rightElbowAngle = calculateAngle(rShoulder, rElbow, rWrist);
    const avgElbowAngle = (leftElbowAngle + rightElbowAngle) / 2;
    
    // Разница в углах коленей
    const kneeDiff = Math.abs(leftKneeAngle - rightKneeAngle);
    
    // Высота ключевых точек для определения положения тела
    const avgShoulderY = (lShoulder.y + rShoulder.y) / 2;
    const avgHipY = (lHip.y + rHip.y) / 2;
    const avgAnkleY = (lAnkle.y + rAnkle.y) / 2;
    
    // Ключевой показатель: разница высот плеч и лодыжек
    const shoulderToAnkleDiff = Math.abs(avgShoulderY - avgAnkleY);
    
    // Определяем горизонтальность и вертикальность тела
    const bodyIsHorizontal = shoulderToAnkleDiff < 0.4; // Меньшая разница = горизонтальнее
    const bodyIsVertical = shoulderToAnkleDiff > 0.6; // Большая разница = вертикальнее
    
    // Разница высоты между левой и правой стороной
    const shoulderHeightDiff = Math.abs(lShoulder.y - rShoulder.y);
    const hipHeightDiff = Math.abs(lHip.y - rHip.y);
    const ankleHeightDiff = Math.abs(lAnkle.y - rAnkle.y);

    // 1. ВЫПАДЫ - асимметрия как главный признак (проверяем первыми)
    const isLunge = 
        // Большая разница в углах коленей (главный признак)
        kneeDiff > 60 &&
        // Одно колено сильно согнуто
        Math.min(leftKneeAngle, rightKneeAngle) < 100 &&
        // Другое колено почти прямо или слегка согнуто
        Math.max(leftKneeAngle, rightKneeAngle) > 140 &&
        // Бедра на разной высоте
        hipHeightDiff > 0.15 &&
        // Лодыжки на разной высоте
        ankleHeightDiff > 0.2 &&
        // Тело вертикально или почти вертикально
        bodyIsVertical;

    // 2. ПЛАНКА - горизонтальность как главный признак
    const isPlank = 
        // Тело должно быть горизонтально (главный признак!)
        bodyIsHorizontal &&
        // Плечи выше бедер (проверка ориентации)
        avgShoulderY < avgHipY &&
        // Ноги почти прямые (колени не сильно согнуты)
        leftKneeAngle > 150 &&
        rightKneeAngle > 150 &&
        // Симметрия (обе стороны на одном уровне)
        shoulderHeightDiff < 0.1 &&
        hipHeightDiff < 0.1 &&
        // Локти могут быть согнуты (отжимание) или прямые
        (avgElbowAngle > 140 || avgElbowAngle < 90); // Допускаем и прямые и согнутые локти

    // 3. ПРИСЕДАНИЯ - симметричное сгибание (проверяем последними)
    const isSquat = 
        // ОБЕ ноги согнуты симметрично (главный признак!)
        leftKneeAngle < 130 &&
        rightKneeAngle < 130 &&
        // Симметрично (разница небольшая)
        kneeDiff < 30 &&
        // Тело вертикально или слегка наклонено вперед
        bodyIsVertical &&
        // Бедра ниже плеч (приседаем вниз)
        avgHipY > avgShoulderY &&
        // Лодыжки примерно на одном уровне (симметрия)
        ankleHeightDiff < 0.15;

    // Определяем с приоритетом: сначала самые уникальные признаки
    if (isLunge) {
        console.log("Определен выпад:", { 
            kneeDiff, leftKneeAngle, rightKneeAngle,
            hipHeightDiff, ankleHeightDiff,
            bodyIsVertical 
        });
        return 'lunges';
    }
    
    if (isPlank) {
        console.log("Определена планка:", { 
            leftKneeAngle, rightKneeAngle, 
            bodyIsHorizontal, shoulderToAnkleDiff,
            shoulderHeightDiff, hipHeightDiff 
        });
        return 'plank';
    }
    
    if (isSquat) {
        console.log("Определен присед:", { 
            leftKneeAngle, rightKneeAngle, kneeDiff,
            avgHipY, avgShoulderY,
            bodyIsVertical 
        });
        return 'squats';
    }

    console.log("Ничего не определено:", { 
        leftKneeAngle, rightKneeAngle, 
        bodyIsHorizontal, bodyIsVertical, shoulderToAnkleDiff,
        kneeDiff, hipHeightDiff 
    });
    return 'none';
}

/**
 * УПРОЩЕННАЯ функция обратной связи
 */
function giveFeedback(exercise, landmarks) {
    if (exercise === 'none') {
        return 'Встаньте в положение упражнения (присед, выпад, планка)';
    }

    const lHip = landmarks[23], lKnee = landmarks[25], lAnkle = landmarks[27];
    const rHip = landmarks[24], rKnee = landmarks[26], rAnkle = landmarks[28];
    
    const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
    const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
    const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;

    switch(exercise) {
        case 'plank':
            if (avgKneeAngle < 170) return "Выпрямите ноги полностью!";
            return "Отличная планка! Держите тело прямо.";
            
        case 'squats':
            if (avgKneeAngle > 100) return "Приседайте глубже!";
            if (avgKneeAngle < 70) return "Не заваливайтесь!";
            return "Хороший присед!";
            
        case 'lunges':
            const kneeDiff = Math.abs(leftKneeAngle - rightKneeAngle);
            if (kneeDiff < 70) return "Сделайте выпад глубже!";
            return "Хороший выпад!";
            
        default:
            return "Продолжайте!";
    }
}

/**
 * Обработка результатов детекции (только для видео)
 */
function processVideoResults(results, timestamp) {
    if (!isPhotoMode) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    }
    
    processLandmarks(results, timestamp);
}

/**
 * Обработка результатов детекции (для фото)
 */
function processPhotoResults(results, timestamp, img) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    
    processLandmarks(results, timestamp);
}

/**
 * Общая обработка landmarks (для видео и фото)
 */
function processLandmarks(results, timestamp) {
    if (results.landmarks?.length > 0) {
        const landmarks = results.landmarks[0];
        const drawingUtils = new DrawingUtils(ctx);
        
        // Рисуем скелет поверх изображения
        drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { 
            color: '#00ff9d', 
            lineWidth: 4 
        });
        drawingUtils.drawLandmarks(landmarks, { 
            color: '#ff3366', 
            radius: 5 
        });

        // Определяем упражнение
        const raw = detectRawExercise(landmarks);
        exerciseHistory[historyIndex] = raw;
        historyIndex = (historyIndex + 1) % HISTORY_LENGTH;
        
        const stableExercise = mostFrequent(exerciseHistory);

        // Обновляем состояние только если упражнение определено
        if (stableExercise !== 'none') {
            lastKnownExerciseTime = timestamp;
            
            // Если упражнение изменилось
            if (stableExercise !== currentExercise) {
                currentExercise = stableExercise;
                repCount = 0;
                plankStartTime = 0;
                squatStage = 'up';
                lungeStage = 'standing';
                repCountEl.textContent = '0';
                timerEl.textContent = '0';
                
                // Обновляем название упражнения
                const names = {
                    squats: 'Приседания',
                    lunges: 'Выпады',
                    plank: 'Планка'
                };
                exerciseNameEl.textContent = names[currentExercise] || 'Упражнение';
                exerciseNameEl.style.color = '#39ff14';
            }
            
            // Обработка специфичных для упражнения действий
            if (currentExercise === 'plank') {
                if (plankStartTime === 0) plankStartTime = timestamp;
                const seconds = Math.floor((timestamp - plankStartTime) / 1000);
                timerEl.textContent = seconds;
            } else {
                timerEl.textContent = '0';
            }
            
            // Счётчик для приседаний
            if (currentExercise === 'squats') {
                const avgKneeAngle = (calculateAngle(landmarks[23], landmarks[25], landmarks[27]) +
                                      calculateAngle(landmarks[24], landmarks[26], landmarks[28])) / 2;
                
                if (squatStage === 'up' && avgKneeAngle < 100) {
                    squatStage = 'down';
                } else if (squatStage === 'down' && avgKneeAngle > 140) {
                    squatStage = 'up';
                    repCount++;
                    repCountEl.textContent = repCount;
                }
            }
            
            // Счётчик для выпадов
            if (currentExercise === 'lunges') {
                const kneeDiff = Math.abs(
                    calculateAngle(landmarks[23], landmarks[25], landmarks[27]) -
                    calculateAngle(landmarks[24], landmarks[26], landmarks[28])
                );
                
                if (lungeStage === 'standing' && kneeDiff > 60) {
                    lungeStage = 'lunge';
                } else if (lungeStage === 'lunge' && kneeDiff < 30) {
                    lungeStage = 'standing';
                    repCount++;
                    repCountEl.textContent = repCount;
                }
            }
            
            feedbackEl.textContent = giveFeedback(currentExercise, landmarks);
            feedbackEl.style.color = "#39ff14";
            
        } else {
            // Если упражнение не определено
            if (timestamp - lastKnownExerciseTime > RESET_AFTER_NONE_MS) {
                currentExercise = 'none';
                exerciseHistory.fill('none');
                historyIndex = 0;
                exerciseNameEl.textContent = 'Определение упражнения...';
                exerciseNameEl.style.color = '#ffcc00';
                feedbackEl.textContent = 'Встаньте в положение для упражнения';
                feedbackEl.style.color = '#ffcc00';
            } else {
                feedbackEl.textContent = 'Упражнение не распознано. Проверьте позу.';
                feedbackEl.style.color = '#ffcc00';
            }
        }
    } else {
        feedbackEl.textContent = 'Человек не найден в кадре';
        feedbackEl.style.color = '#ff4757';
    }
}

/**
 * Цикл обработки видео
 */
function runVideoDetection() {
    if (!currentPoseLandmarker || isPhotoMode) return;
    const now = performance.now();
    const results = currentPoseLandmarker.detectForVideo(video, now);
    processVideoResults(results, now);
    requestAnimationFrame(runVideoDetection);
}

// -----------------------
// Кнопка запуска камеры
// -----------------------
document.getElementById('startButton').addEventListener('click', async () => {
    try {
        isPhotoMode = false;
        currentPoseLandmarker = await initPoseLandmarkerVideo();
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: "user",
                width: { ideal: 640 },
                height: { ideal: 480 }
            } 
        });
        
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            video.play();
            runVideoDetection();
            feedbackEl.textContent = 'Камера запущена. Встаньте в положение упражнения';
            feedbackEl.style.color = '#39ff14';
        };
        
    } catch (err) {
        feedbackEl.textContent = "Ошибка доступа к камере: " + err.message;
        feedbackEl.style.color = '#ff4757';
        console.error(err);
    }
});

// -----------------------
// Кнопка анализа фотографии
// -----------------------
document.getElementById('analyzePhotoButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('photoUpload');
    if (!fileInput.files?.length) {
        feedbackEl.textContent = 'Выберите фото!';
        feedbackEl.style.color = '#ff4757';
        return;
    }

    isPhotoMode = true;
    currentPoseLandmarker = await initPoseLandmarkerImage();

    const file = fileInput.files[0];
    const img = new Image();
    
    img.onload = async () => {
        try {
            canvas.width = img.width;
            canvas.height = img.height;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            const results = await currentPoseLandmarker.detect(img);
            
            processPhotoResults(results, performance.now(), img);
            
            if (results.landmarks?.length > 0) {
                feedbackEl.textContent = 'Фото проанализировано! Упражнение определено.';
                feedbackEl.style.color = '#39ff14';
            } else {
                feedbackEl.textContent = 'На фото не обнаружен человек.';
                feedbackEl.style.color = '#ff4757';
            }
            
        } catch (e) {
            console.error('Ошибка анализа фото:', e);
            feedbackEl.textContent = 'Ошибка анализа фото: ' + e.message;
            feedbackEl.style.color = '#ff4757';
        }
    };
    
    img.onerror = () => {
        feedbackEl.textContent = 'Не удалось загрузить изображение';
        feedbackEl.style.color = '#ff4757';
    };
    
    img.src = URL.createObjectURL(file);
});