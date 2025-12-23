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
const HISTORY_LENGTH = 5;
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
 * ПРАВИЛЬНАЯ функция определения упражнения
 */
function detectRawExercise(landmarks) {
    if (!landmarks || landmarks.length < 29) {
        return 'none';
    }

    const lShoulder = landmarks[11], rShoulder = landmarks[12];
    const lElbow = landmarks[13], rElbow = landmarks[14];
    const lWrist = landmarks[15], rWrist = landmarks[16];
    const lHip = landmarks[23], rHip = landmarks[24];
    const lKnee = landmarks[25], rKnee = landmarks[26];
    const lAnkle = landmarks[27], rAnkle = landmarks[28];

    // Вычисляем ключевые углы
    const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
    const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
    const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;
    
    // Разница в углах коленей
    const kneeDiff = Math.abs(leftKneeAngle - rightKneeAngle);
    
    // Высота ключевых точек
    const avgShoulderY = (lShoulder.y + rShoulder.y) / 2;
    const avgHipY = (lHip.y + rHip.y) / 2;
    const avgAnkleY = (lAnkle.y + rAnkle.y) / 2;
    
    // Разница высот между левой и правой стороной
    const shoulderHeightDiff = Math.abs(lShoulder.y - rShoulder.y);
    const hipHeightDiff = Math.abs(lHip.y - rHip.y);
    const ankleHeightDiff = Math.abs(lAnkle.y - rAnkle.y);
    
    // Горизонтальность тела (разница высот плеч и лодыжек)
    const verticalBodyDiff = Math.abs(avgShoulderY - avgAnkleY);
    
    // Углы в локтях
    const leftElbowAngle = calculateAngle(lShoulder, lElbow, lWrist);
    const rightElbowAngle = calculateAngle(rShoulder, rElbow, rWrist);
    const avgElbowAngle = (leftElbowAngle + rightElbowAngle) / 2;

    console.log("=== ДАННЫЕ ДЛЯ ОПРЕДЕЛЕНИЯ ===");
    console.log("Углы коленей: Л=" + leftKneeAngle.toFixed(0) + "°, П=" + rightKneeAngle.toFixed(0) + "°");
    console.log("Разница углов: " + kneeDiff.toFixed(0) + "°");
    console.log("Вертикальная разница тела: " + verticalBodyDiff.toFixed(2));
    console.log("Высота плеч: " + avgShoulderY.toFixed(2) + ", бедер: " + avgHipY.toFixed(2));

    // 1. ПЛАНКА - тело горизонтально + ноги прямые
    if (verticalBodyDiff < 0.25 && 
        leftKneeAngle > 150 && 
        rightKneeAngle > 150 &&
        avgShoulderY < avgHipY + 0.1) { // Плечи не сильно ниже бедер
        console.log("✅ ОПРЕДЕЛЕНА: ПЛАНКА (горизонтально: " + verticalBodyDiff.toFixed(2) + ")");
        return 'plank';
    }
    
    // 2. ВЫПАДЫ - большая разница в углах коленей + асимметрия
    if (kneeDiff > 55 && 
        Math.min(leftKneeAngle, rightKneeAngle) < 105 &&
        Math.max(leftKneeAngle, rightKneeAngle) > 145 &&
        hipHeightDiff > 0.08 &&
        verticalBodyDiff > 0.3) {
        console.log("✅ ОПРЕДЕЛЕН: ВЫПАД (разница коленей: " + kneeDiff.toFixed(0) + "°)");
        return 'lunges';
    }
    
    // 3. ПРИСЕДАНИЯ - оба колена согнуты + симметрично
    if (leftKneeAngle < 135 && 
        rightKneeAngle < 135 && 
        kneeDiff < 35 &&
        verticalBodyDiff > 0.35 &&
        avgHipY > avgShoulderY + 0.05) { // Бедра ниже плеч
        console.log("✅ ОПРЕДЕЛЕН: ПРИСЕД (оба колена согнуты)");
        return 'squats';
    }
    
    console.log("❌ НЕ ОПРЕДЕЛЕНО");
    return 'none';
}

/**
 * ПОНЯТНАЯ функция обратной связи
 */
function giveFeedback(exercise, landmarks) {
    if (exercise === 'none') {
        return '🏃 Встаньте в положение упражнения (присед, выпад, планка)';
    }

    const lShoulder = landmarks[11], rShoulder = landmarks[12];
    const lHip = landmarks[23], rHip = landmarks[24];
    const lKnee = landmarks[25], rKnee = landmarks[26];
    const lAnkle = landmarks[27], rAnkle = landmarks[28];
    
    // Вычисляем углы
    const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
    const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
    const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;
    const kneeDiff = Math.abs(leftKneeAngle - rightKneeAngle);
    
    // Высота для проверки ровности
    const shoulderHeightDiff = Math.abs(lShoulder.y - rShoulder.y);
    const hipHeightDiff = Math.abs(lHip.y - rHip.y);

    switch(exercise) {
        case 'plank':
            if (avgKneeAngle < 165) {
                return "📝 Для планки: Выпрямите ноги! Колени должны быть прямыми";
            }
            if (shoulderHeightDiff > 0.08) {
                return "📝 Для планки: Выровняйте плечи! Они должны быть на одной линии";
            }
            return "✅ Идеальная планка! Тело образует прямую линию";
            
        case 'squats':
            if (avgKneeAngle > 110) {
                return "📝 Для приседа: Присядьте глубже! Цель - 90 градусов в коленях";
            }
            if (avgKneeAngle < 75) {
                return "📝 Для приседа: Слишком глубоко! Колени не должны болеть";
            }
            if (kneeDiff > 20) {
                return "📝 Для приседа: Выровняйте колени! Делайте симметрично";
            }
            if (hipHeightDiff > 0.1) {
                return "📝 Для приседа: Выровняйте бедра!";
            }
            return "✅ Идеальный присед! Отличная техника";
            
        case 'lunges':
            if (kneeDiff < 60) {
                return "📝 Для выпада: Сделайте шаг шире! Разница в коленях должна быть больше";
            }
            const minKneeAngle = Math.min(leftKneeAngle, rightKneeAngle);
            if (minKneeAngle > 95) {
                return "📝 Для выпада: Согните переднее колено сильнее! Цель - 90 градусов";
            }
            return "✅ Идеальный выпад! Хорошая амплитуда и баланс";
            
        default:
            return "💪 Продолжайте в том же духе!";
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
        console.log("Сырое определение: " + raw);
        
        exerciseHistory[historyIndex] = raw;
        historyIndex = (historyIndex + 1) % HISTORY_LENGTH;
        
        const stableExercise = mostFrequent(exerciseHistory);
        console.log("Стабильное определение: " + stableExercise);

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
                    squats: '🏋️ ПРИСЕДАНИЯ',
                    lunges: '🦵 ВЫПАДЫ',
                    plank: '🧘‍♂️ ПЛАНКА'
                };
                exerciseNameEl.textContent = names[currentExercise] || '🤔 Упражнение';
                exerciseNameEl.style.color = '#39ff14';
                console.log("✨ УПРАЖНЕНИЕ ИЗМЕНИЛОСЬ НА: " + currentExercise);
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
                    console.log("⬇️ Присед: опускаемся");
                } else if (squatStage === 'down' && avgKneeAngle > 130) {
                    squatStage = 'up';
                    repCount++;
                    repCountEl.textContent = repCount;
                    console.log("⬆️ Присед: поднимаемся, повтор: " + repCount);
                }
            }
            
            // Счётчик для выпадов
            if (currentExercise === 'lunges') {
                const leftKneeAngle = calculateAngle(landmarks[23], landmarks[25], landmarks[27]);
                const rightKneeAngle = calculateAngle(landmarks[24], landmarks[26], landmarks[28]);
                const kneeDiff = Math.abs(leftKneeAngle - rightKneeAngle);
                
                if (lungeStage === 'standing' && kneeDiff > 60) {
                    lungeStage = 'lunge';
                    console.log("⬇️ Выпад: опускаемся");
                } else if (lungeStage === 'lunge' && kneeDiff < 40) {
                    lungeStage = 'standing';
                    repCount++;
                    repCountEl.textContent = repCount;
                    console.log("⬆️ Выпад: поднимаемся, повтор: " + repCount);
                }
            }
            
            // Обратная связь
            const feedback = giveFeedback(currentExercise, landmarks);
            feedbackEl.textContent = feedback;
            feedbackEl.style.color = "#39ff14";
            
            console.log("💬 Обратная связь: " + feedback);
            
        } else {
            // Если упражнение не определено
            if (timestamp - lastKnownExerciseTime > RESET_AFTER_NONE_MS) {
                currentExercise = 'none';
                exerciseHistory.fill('none');
                historyIndex = 0;
                exerciseNameEl.textContent = '🔍 Определение упражнения...';
                exerciseNameEl.style.color = '#ffcc00';
                feedbackEl.textContent = 'Встаньте в положение для упражнения';
                feedbackEl.style.color = '#ffcc00';
                console.log("🔄 Сброс: упражнение не определено");
            } else {
                feedbackEl.textContent = 'Упражнение не распознано. Проверьте позу.';
                feedbackEl.style.color = '#ffcc00';
            }
        }
    } else {
        feedbackEl.textContent = '👤 Человек не найден в кадре';
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
            feedbackEl.textContent = '📹 Камера запущена. Встаньте в положение упражнения';
            feedbackEl.style.color = '#39ff14';
            console.log("🎥 Камера запущена, разрешение: " + video.videoWidth + "x" + video.videoHeight);
        };
        
    } catch (err) {
        feedbackEl.textContent = "❌ Ошибка доступа к камере: " + err.message;
        feedbackEl.style.color = '#ff4757';
        console.error("Ошибка камеры:", err);
    }
});

// -----------------------
// Кнопка анализа фотографии
// -----------------------
document.getElementById('analyzePhotoButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('photoUpload');
    if (!fileInput.files?.length) {
        feedbackEl.textContent = '📷 Выберите фото!';
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
            
            console.log("📸 Анализируем фото: " + img.width + "x" + img.height);
            const results = await currentPoseLandmarker.detect(img);
            
            processPhotoResults(results, performance.now(), img);
            
            if (results.landmarks?.length > 0) {
                feedbackEl.textContent = '✅ Фото проанализировано! Упражнение определено.';
                feedbackEl.style.color = '#39ff14';
            } else {
                feedbackEl.textContent = '❌ На фото не обнаружен человек.';
                feedbackEl.style.color = '#ff4757';
            }
            
        } catch (e) {
            console.error('Ошибка анализа фото:', e);
            feedbackEl.textContent = '❌ Ошибка анализа фото: ' + e.message;
            feedbackEl.style.color = '#ff4757';
        }
    };
    
    img.onerror = () => {
        feedbackEl.textContent = '❌ Не удалось загрузить изображение';
        feedbackEl.style.color = '#ff4757';
    };
    
    img.src = URL.createObjectURL(file);
});

// -----------------------
// Кнопка сброса
// -----------------------
if (!document.getElementById('resetButton')) {
    const resetButton = document.createElement('button');
    resetButton.id = 'resetButton';
    resetButton.textContent = '🔄 Сбросить счетчики';
    resetButton.style.cssText = `
        background-color: #ff4757;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin: 10px;
        font-weight: bold;
        font-size: 16px;
    `;
    document.querySelector('.container').appendChild(resetButton);
    
    resetButton.addEventListener('click', () => {
        repCount = 0;
        plankStartTime = 0;
        repCountEl.textContent = '0';
        timerEl.textContent = '0';
        feedbackEl.textContent = '✅ Счетчики сброшены';
        feedbackEl.style.color = '#39ff14';
        console.log("🔄 Счетчики сброшены");
    });
}