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

// Счётчик повторений для выпадов
let lungeRepCount = 0;
let lastLungeState = 'standing';

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
 * УЛУЧШЕННАЯ функция определения упражнения с четкими критериями
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

    // Вычисляем ключевые углы
    const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
    const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
    
    // Углы в локтях
    const leftElbowAngle = calculateAngle(lShoulder, lElbow, lWrist);
    const rightElbowAngle = calculateAngle(rShoulder, rElbow, rWrist);
    const avgElbowAngle = (leftElbowAngle + rightElbowAngle) / 2;
    
    // Разница в углах коленей - КЛЮЧЕВОЙ параметр для выпадов
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
    const shoulderToAnkleDiff = Math.abs(avgShoulderY - avgAnkleY);
    
    // Вертикальность (разница высот плеч и бедер)
    const shoulderToHipDiff = Math.abs(avgShoulderY - avgHipY);

    // 1. ПЛАНКА - самое простое для определения
    const isPlank = (
        // Тело горизонтально (плечи и лодыжки примерно на одной высоте)
        shoulderToAnkleDiff < 0.3 &&
        // Ноги прямые или почти прямые
        leftKneeAngle > 150 &&
        rightKneeAngle > 150 &&
        // Локти согнуты (планка на локтях) или прямые (планка на руках)
        (avgElbowAngle < 100 || avgElbowAngle > 150) &&
        // Плечи выше бедер (правильная ориентация)
        avgShoulderY < avgHipY + 0.2
    );

    // 2. ВЫПАДЫ - главный признак: БОЛЬШАЯ РАЗНИЦА в углах коленей
    const isLunge = (
        // КЛЮЧЕВОЙ ПРИЗНАК: одно колено сильно согнуто, другое почти прямо
        (leftKneeAngle < 100 && rightKneeAngle > 140) ||
        (rightKneeAngle < 100 && leftKneeAngle > 140) &&
        // Разница углов большая
        kneeDiff > 50 &&
        // Бедра на разной высоте
        hipHeightDiff > 0.1 &&
        // Тело вертикально или почти вертикально
        shoulderToAnkleDiff > 0.4
    );

    // 3. ПРИСЕДАНИЯ - оба колена согнуты примерно одинаково
    const isSquat = (
        // Оба колена согнуты
        leftKneeAngle < 140 &&
        rightKneeAngle < 140 &&
        // Колени согнуты примерно одинаково (симметрия)
        kneeDiff < 40 &&
        // Бедра ниже плеч (мы приседаем вниз)
        avgHipY > avgShoulderY + 0.1 &&
        // Тело вертикально
        shoulderToAnkleDiff > 0.5
    );

    // Приоритет проверки: выпады -> планка -> приседания
    if (isLunge) {
        console.log("🔥 ВЫПАД ОПРЕДЕЛЕН! Углы коленей: " + 
                   leftKneeAngle.toFixed(0) + "° / " + rightKneeAngle.toFixed(0) + 
                   "°, разница: " + kneeDiff.toFixed(0) + "°");
        return 'lunges';
    }
    
    if (isPlank) {
        console.log("🔥 ПЛАНКА ОПРЕДЕЛЕНА! Горизонтальность: " + 
                   shoulderToAnkleDiff.toFixed(2) + ", колени: " + 
                   leftKneeAngle.toFixed(0) + "° / " + rightKneeAngle.toFixed(0) + "°");
        return 'plank';
    }
    
    if (isSquat) {
        console.log("🔥 ПРИСЕД ОПРЕДЕЛЕН! Оба колена согнуты: " + 
                   leftKneeAngle.toFixed(0) + "° / " + rightKneeAngle.toFixed(0) + "°");
        return 'squats';
    }

    console.log("❌ Ничего не определено");
    return 'none';
}

/**
 * ПОДРОБНАЯ функция обратной связи с советами по технике
 */
function giveFeedback(exercise, landmarks) {
    if (exercise === 'none') {
        return '🏃‍♂️ Встаньте в положение для упражнения: присед, выпад или планка';
    }

    const lShoulder = landmarks[11], rShoulder = landmarks[12];
    const lElbow = landmarks[13], rElbow = landmarks[14];
    const lHip = landmarks[23], rHip = landmarks[24];
    const lKnee = landmarks[25], rKnee = landmarks[26];
    const lAnkle = landmarks[27], rAnkle = landmarks[28];
    
    // Вычисляем углы
    const leftKneeAngle = calculateAngle(lHip, lKnee, lAnkle);
    const rightKneeAngle = calculateAngle(rHip, rKnee, rAnkle);
    const avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2;
    const kneeDiff = Math.abs(leftKneeAngle - rightKneeAngle);
    
    // Углы в бедрах (для проверки спины)
    const leftHipAngle = calculateAngle(lShoulder, lHip, lKnee);
    const rightHipAngle = calculateAngle(rShoulder, rHip, rKnee);
    const avgHipAngle = (leftHipAngle + rightHipAngle) / 2;
    
    // Углы в локтях
    const leftElbowAngle = calculateAngle(lShoulder, lElbow, lWrist);
    const rightElbowAngle = calculateAngle(rShoulder, rElbow, rWrist);
    const avgElbowAngle = (leftElbowAngle + rightElbowAngle) / 2;
    
    // Проверка симметрии
    const shoulderHeightDiff = Math.abs(lShoulder.y - rShoulder.y);
    const hipHeightDiff = Math.abs(lHip.y - rHip.y);

    switch(exercise) {
        case 'plank':
            const feedbackPlank = [];
            
            // 1. Проверка ног
            if (avgKneeAngle < 170) {
                feedbackPlank.push("Ноги должны быть прямыми!");
            }
            
            // 2. Проверка бедер
            if (avgHipAngle > 190 || avgHipAngle < 170) {
                feedbackPlank.push("Бедра должны быть на одной линии с плечами!");
            }
            
            // 3. Проверка локтей
            if (avgElbowAngle > 160) {
                feedbackPlank.push("Для планки на локтях: согните локти под 90°!");
            }
            
            // 4. Проверка симметрии
            if (shoulderHeightDiff > 0.08 || hipHeightDiff > 0.08) {
                feedbackPlank.push("Выровняйте плечи и бедра!");
            }
            
            if (feedbackPlank.length === 0) {
                return "✅ ИДЕАЛЬНАЯ ПЛАНКА! Советы: дышите ровно, напрягите пресс, не опускайте голову";
            } else {
                return "📝 КОРРЕКТИРОВКИ: " + feedbackPlank.join(" ") + 
                       " | Совет: держите тело прямой линией";
            }
            
        case 'squats':
            const feedbackSquat = [];
            
            // 1. Глубина приседа
            if (avgKneeAngle > 110) {
                feedbackSquat.push("Приседайте глубже! Колени должны сгибаться под 90°");
            } else if (avgKneeAngle < 70) {
                feedbackSquat.push("Не заваливайтесь! Слишком глубокий присед вреден для коленей");
            }
            
            // 2. Симметрия
            if (kneeDiff > 15) {
                feedbackSquat.push("Выровняйте колени! Приседайте симметрично");
            }
            
            // 3. Спина
            if (avgHipAngle < 140) {
                feedbackSquat.push("Держите спину прямой! Не наклоняйтесь сильно вперед");
            }
            
            // 4. Бедра
            if (hipHeightDiff > 0.1) {
                feedbackSquat.push("Бедра должны быть на одном уровне!");
            }
            
            if (feedbackSquat.length === 0) {
                return "✅ ИДЕАЛЬНЫЙ ПРИСЕД! Советы: колени над стопами, грудь вперед, пятки не отрывать";
            } else {
                return "📝 КОРРЕКТИРОВКИ: " + feedbackSquat.join(" ") + 
                       " | Совет: колени не должны выходить за носки";
            }
            
        case 'lunges':
            const feedbackLunge = [];
            
            // Определяем, какая нога впереди
            const isLeftForward = leftKneeAngle < rightKneeAngle;
            const frontKneeAngle = isLeftForward ? leftKneeAngle : rightKneeAngle;
            const backKneeAngle = isLeftForward ? rightKneeAngle : leftKneeAngle;
            const frontKnee = isLeftForward ? lKnee : rKnee;
            const frontAnkle = isLeftForward ? lAnkle : rAnkle;
            
            // 1. Угол переднего колена
            if (frontKneeAngle > 95) {
                feedbackLunge.push("Согните переднее колено сильнее! Цель - 90°");
            } else if (frontKneeAngle < 70) {
                feedbackLunge.push("Переднее колено слишком согнуто!");
            }
            
            // 2. Заднее колено
            if (backKneeAngle < 150) {
                feedbackLunge.push("Заднее колено должно быть почти прямым!");
            }
            
            // 3. Проверка колено-носок
            const kneeOverToe = Math.abs(frontKnee.x - frontAnkle.x) > 0.15;
            if (kneeOverToe) {
                feedbackLunge.push("Колено не должно выходить за носок!");
            }
            
            // 4. Корпус
            if (avgHipAngle < 160) {
                feedbackLunge.push("Держите корпус вертикально! Не наклоняйтесь вперед");
            }
            
            // 5. Глубина выпада
            if (kneeDiff < 60) {
                feedbackLunge.push("Шаг должен быть шире для лучшей амплитуды");
            }
            
            if (feedbackLunge.length === 0) {
                return "✅ ИДЕАЛЬНЫЙ ВЫПАД! Советы: шаг достаточно широкий, корпус вертикальный, переднее колено 90°";
            } else {
                return "📝 КОРРЕКТИРОВКИ: " + feedbackLunge.join(" ") + 
                       " | Совет: вес равномерно распределен между ногами";
            }
            
        default:
            return "💪 Продолжайте выполнять упражнение!";
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
        
        // Стабилизация
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
                lungeRepCount = 0;
                repCountEl.textContent = '0';
                timerEl.textContent = '0';
                
                // Обновляем название упражнения
                const names = {
                    squats: '🏋️‍♂️ ПРИСЕДАНИЯ',
                    lunges: '🦵 ВЫПАДЫ',
                    plank: '🧘‍♂️ ПЛАНКА'
                };
                exerciseNameEl.textContent = names[currentExercise] || '🤔 Упражнение';
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
                } else if (squatStage === 'down' && avgKneeAngle > 130) {
                    squatStage = 'up';
                    repCount++;
                    repCountEl.textContent = repCount;
                }
            }
            
            // Счётчик для выпадов - УЛУЧШЕННЫЙ алгоритм
            if (currentExercise === 'lunges') {
                const leftKneeAngle = calculateAngle(landmarks[23], landmarks[25], landmarks[27]);
                const rightKneeAngle = calculateAngle(landmarks[24], landmarks[26], landmarks[28]);
                const kneeDiff = Math.abs(leftKneeAngle - rightKneeAngle);
                
                // Определяем фазу выпада
                if (kneeDiff > 70 && lungeStage === 'standing') {
                    lungeStage = 'down';
                    console.log("⬇️ Выпад: опускаемся вниз");
                } else if (kneeDiff < 50 && lungeStage === 'down') {
                    lungeStage = 'standing';
                    repCount++;
                    repCountEl.textContent = repCount;
                    console.log("⬆️ Выпад: поднимаемся, повтор: " + repCount);
                }
            }
            
            // ВСЕГДА показываем обратную связь!
            const feedback = giveFeedback(currentExercise, landmarks);
            feedbackEl.innerHTML = feedback;
            feedbackEl.style.color = "#39ff14";
            feedbackEl.style.fontSize = "18px";
            feedbackEl.style.padding = "10px";
            feedbackEl.style.backgroundColor = "rgba(0, 0, 0, 0.7)";
            feedbackEl.style.borderRadius = "5px";
            
        } else {
            // Если упражнение не определено
            if (timestamp - lastKnownExerciseTime > RESET_AFTER_NONE_MS) {
                currentExercise = 'none';
                exerciseHistory.fill('none');
                historyIndex = 0;
                exerciseNameEl.textContent = '🔍 Определение упражнения...';
                exerciseNameEl.style.color = '#ffcc00';
                feedbackEl.textContent = '🏃‍♂️ Встаньте в положение для упражнения (присед, выпад, планка)';
                feedbackEl.style.color = '#ffcc00';
            } else {
                feedbackEl.textContent = '🤔 Упражнение не распознано. Проверьте позу и освещение.';
                feedbackEl.style.color = '#ffcc00';
            }
        }
    } else {
        feedbackEl.innerHTML = '👤 <strong>Человек не найден в кадре</strong><br>Убедитесь, что вы в поле зрения камеры';
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
            feedbackEl.innerHTML = '📹 <strong>Камера запущена!</strong><br>Встаньте в положение для упражнения';
            feedbackEl.style.color = '#39ff14';
        };
        
    } catch (err) {
        feedbackEl.innerHTML = '❌ <strong>Ошибка доступа к камере</strong><br>' + err.message;
        feedbackEl.style.color = '#ff4757';
    }
});

// -----------------------
// Кнопка анализа фотографии
// -----------------------
document.getElementById('analyzePhotoButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('photoUpload');
    if (!fileInput.files?.length) {
        feedbackEl.innerHTML = '📷 <strong>Выберите фото!</strong>';
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
                feedbackEl.innerHTML = '✅ <strong>Фото проанализировано!</strong><br>Упражнение определено';
                feedbackEl.style.color = '#39ff14';
            } else {
                feedbackEl.innerHTML = '❌ <strong>На фото не обнаружен человек</strong>';
                feedbackEl.style.color = '#ff4757';
            }
            
        } catch (e) {
            feedbackEl.innerHTML = '❌ <strong>Ошибка анализа фото</strong><br>' + e.message;
            feedbackEl.style.color = '#ff4757';
        }
    };
    
    img.onerror = () => {
        feedbackEl.innerHTML = '❌ <strong>Не удалось загрузить изображение</strong>';
        feedbackEl.style.color = '#ff4757';
    };
    
    img.src = URL.createObjectURL(file);
});

// Добавляем стили для feedback элемента
document.addEventListener('DOMContentLoaded', () => {
    if (feedbackEl) {
        feedbackEl.style.cssText = `
            font-size: 18px;
            font-weight: bold;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            background-color: rgba(0, 0, 0, 0.7);
            color: #39ff14;
            min-height: 80px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            line-height: 1.4;
        `;
    }
});