package com.example.arborscan_app

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageButton
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.ar.core.Anchor
import com.google.ar.core.Frame
import com.google.ar.core.HitResult
import com.google.ar.core.Plane
import com.google.ar.core.Point
import com.google.ar.core.Pose
import com.google.ar.core.TrackingState
import com.google.ar.sceneform.AnchorNode
import com.google.ar.sceneform.FrameTime
import com.google.ar.sceneform.Node
import com.google.ar.sceneform.math.Vector3
import com.google.ar.sceneform.rendering.Color as SceneColor
import com.google.ar.sceneform.rendering.MaterialFactory
import com.google.ar.sceneform.rendering.ShapeFactory
import com.google.ar.sceneform.ux.ArFragment
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.sqrt
import kotlin.math.tan
import kotlin.math.PI

class ArMeasureActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_RESULT_JSON = "result_json"
    }

    // Этапы измерения: добавлен FALL_ZONE
    enum class MeasureStep {
        BASE, TOP, CROWN_LEFT, CROWN_RIGHT, TRUNK_LEFT, TRUNK_RIGHT, FALL_ZONE, DONE
    }

    private lateinit var arFragment: ArFragment
    private lateinit var tvStep: TextView
    private lateinit var tvHint: TextView
    private lateinit var tvStatus: TextView
    private lateinit var tvRealtime: TextView
    private lateinit var btnPlace: Button
    private lateinit var btnUndo: ImageButton

    // Сохраненные данные
    private var currentStep = MeasureStep.BASE
    private var baseAnchorNode: AnchorNode? = null
    private var fallZoneNode: Node? = null
    
    // Результаты измерений
    private var finalHeight = 0.0
    private var finalCrownWidth = 0.0
    private var finalTrunkDiameter = 0.0
    
    // Промежуточные углы (в радианах)
    private var crownLeftYaw = 0.0
    private var trunkLeftYaw = 0.0

    // Фоновая ML-модель
    private lateinit var tflite: TFLiteHelper
    private lateinit var treeDetector: TreeDetector
    private var lastMlTime = 0L
    private val mlExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ar_measure)

        arFragment = supportFragmentManager.findFragmentById(R.id.arFragment) as ArFragment
        tvStep = findViewById(R.id.tvStep)
        tvHint = findViewById(R.id.tvHint)
        tvStatus = findViewById(R.id.tvStatus)
        tvRealtime = findViewById(R.id.tvRealtime)
        btnPlace = findViewById(R.id.btnPlace)
        btnUndo = findViewById(R.id.btnUndo)

        arFragment.arSceneView.scene.addOnUpdateListener(this::onSceneUpdate)
        arFragment.setOnTapArPlaneListener { _, _, _ -> } // Отключаем стандартные тапы

        btnPlace.setOnClickListener { onPlaceClicked() }
        btnUndo.setOnClickListener { undoStep() }

        tflite = TFLiteHelper(this)
        treeDetector = TreeDetector(tflite)

        updateUi()
    }

    private fun onSceneUpdate(frameTime: FrameTime) {
        val frame = arFragment.arSceneView.arFrame ?: return

        // Фоновый запуск ML
        val now = System.currentTimeMillis()
        if (now - lastMlTime > 500L) {
            processFrame(frame)
            lastMlTime = now
        }

        if (frame.camera.trackingState != TrackingState.TRACKING) {
            tvStatus.text = "Трекинг AR потерян. Двигайте телефон медленно."
            return
        }

        // Логика реального времени в зависимости от шага
        if (currentStep == MeasureStep.BASE) {
            val hit = getCenterHit(frame)
            if (hit != null) {
                tvStatus.text = "Плоскость найдена. Наведите на корни и нажмите кнопку."
                btnPlace.isEnabled = true
            } else {
                tvStatus.text = "Наведите камеру на текстуру (землю/траву)."
                btnPlace.isEnabled = false
            }
        } else {
            // AR Anchor уже установлен. Считаем углы и дистанцию.
            btnPlace.isEnabled = true
            updateRealtimeMath(frame.camera.pose)
        }
    }

    private fun updateRealtimeMath(cameraPose: Pose) {
        try {
            val baseAnchor = baseAnchorNode?.anchor ?: return
            val distance = getHorizontalDistance(cameraPose, baseAnchor.pose)
            val pitch = getPitch(cameraPose)
            val yaw = getYaw(cameraPose)

            tvRealtime.visibility = View.VISIBLE

            when (currentStep) {
                MeasureStep.TOP -> {
                    val height = calculateHeight(cameraPose, baseAnchor.pose, pitch, distance)
                    tvRealtime.text = "Высота: %.1f м".format(height)
                }
                MeasureStep.CROWN_LEFT -> {
                    tvRealtime.text = "Дистанция: %.1f м".format(distance)
                }
                MeasureStep.CROWN_RIGHT -> {
                    val diff = getAngleDiff(crownLeftYaw, yaw)
                    val width = 2 * distance * tan(diff / 2)
                    tvRealtime.text = "Крона: %.1f м".format(width)
                }
                MeasureStep.TRUNK_LEFT -> {
                    tvRealtime.text = "Дистанция: %.1f м".format(distance)
                }
                MeasureStep.TRUNK_RIGHT -> {
                    val diff = getAngleDiff(trunkLeftYaw, yaw)
                    val trunk = 2 * distance * tan(diff / 2)
                    tvRealtime.text = "Ствол: %.2f м".format(trunk)
                }
                MeasureStep.FALL_ZONE -> {
                    tvRealtime.text = "Зона поражения: R = %.1f м".format(finalHeight)
                }
                else -> tvRealtime.visibility = View.GONE
            }
        } catch (e: Exception) {
            // Игнорируем ошибки при отрисовке UI
        }
    }

    private fun onPlaceClicked() {
        try {
            vibrate() 
            val frame = arFragment.arSceneView.arFrame ?: return
            val cameraPose = frame.camera.pose

            when (currentStep) {
                MeasureStep.BASE -> {
                    val hit = getCenterHit(frame) ?: return
                    val anchor = hit.createAnchor()
                    
                    baseAnchorNode = AnchorNode(anchor).apply {
                        setParent(arFragment.arSceneView.scene)
                        // Рисуем зеленый шарик у основания дерева
                        MaterialFactory.makeOpaqueWithColor(this@ArMeasureActivity, SceneColor(0.27f, 0.88f, 0.63f))
                            .thenAccept { material ->
                                renderable = ShapeFactory.makeSphere(0.05f, Vector3.zero(), material)
                            }
                    }
                    currentStep = MeasureStep.TOP
                }
                MeasureStep.TOP -> {
                    val base = baseAnchorNode!!.anchor!!.pose
                    val distance = getHorizontalDistance(cameraPose, base)
                    finalHeight = calculateHeight(cameraPose, base, getPitch(cameraPose), distance)
                    currentStep = MeasureStep.CROWN_LEFT
                }
                MeasureStep.CROWN_LEFT -> {
                    crownLeftYaw = getYaw(cameraPose)
                    currentStep = MeasureStep.CROWN_RIGHT
                }
                MeasureStep.CROWN_RIGHT -> {
                    val distance = getHorizontalDistance(cameraPose, baseAnchorNode!!.anchor!!.pose)
                    val diff = getAngleDiff(crownLeftYaw, getYaw(cameraPose))
                    finalCrownWidth = 2 * distance * tan(diff / 2)
                    currentStep = MeasureStep.TRUNK_LEFT
                }
                MeasureStep.TRUNK_LEFT -> {
                    trunkLeftYaw = getYaw(cameraPose)
                    currentStep = MeasureStep.TRUNK_RIGHT
                }
                MeasureStep.TRUNK_RIGHT -> {
                    val distance = getHorizontalDistance(cameraPose, baseAnchorNode!!.anchor!!.pose)
                    val diff = getAngleDiff(trunkLeftYaw, getYaw(cameraPose))
                    finalTrunkDiameter = 2 * distance * tan(diff / 2)
                    
                    // Переходим к отрисовке Зоны падения
                    currentStep = MeasureStep.FALL_ZONE
                    drawFallZone(finalHeight)
                }
                MeasureStep.FALL_ZONE -> {
                    // Все шаги и осмотр завершены, сохраняем и выходим
                    currentStep = MeasureStep.DONE
                    finishMeasure()
                    return
                }
                MeasureStep.DONE -> return
            }
            updateUi()
            
        } catch (e: Exception) {
            Log.e("ArMeasureActivity", "Error placing point", e)
            tvStatus.text = "Ошибка фиксации. Попробуйте еще раз."
        }
    }

    private fun drawFallZone(radiusMeters: Double) {
        if (radiusMeters <= 0.0 || baseAnchorNode == null) return

        val r = radiusMeters.toFloat()

        // Создаем полупрозрачный красный материал
        MaterialFactory.makeTransparentWithColor(this, SceneColor(1.0f, 0.2f, 0.2f, 0.4f))
            .thenAccept { material ->
                // Создаем очень плоский цилиндр (по сути круг на земле)
                val cylinder = ShapeFactory.makeCylinder(r, 0.02f, Vector3(0f, 0.01f, 0f), material)
                
                fallZoneNode = Node().apply {
                    setParent(baseAnchorNode)
                    renderable = cylinder
                }
            }
    }

    private fun undoStep() {
        try {
            vibrate()
            when (currentStep) {
                MeasureStep.TOP -> {
                    baseAnchorNode?.anchor?.detach()
                    baseAnchorNode?.setParent(null)
                    baseAnchorNode = null
                    currentStep = MeasureStep.BASE
                }
                MeasureStep.CROWN_LEFT -> currentStep = MeasureStep.TOP
                MeasureStep.CROWN_RIGHT -> currentStep = MeasureStep.CROWN_LEFT
                MeasureStep.TRUNK_LEFT -> currentStep = MeasureStep.CROWN_RIGHT
                MeasureStep.TRUNK_RIGHT -> currentStep = MeasureStep.TRUNK_LEFT
                MeasureStep.FALL_ZONE -> {
                    fallZoneNode?.setParent(null)
                    fallZoneNode = null
                    currentStep = MeasureStep.TRUNK_RIGHT
                }
                else -> {}
            }
            updateUi()
        } catch (e: Exception) {
            Log.e("ArMeasureActivity", "Error undoing step", e)
        }
    }

    private fun updateUi() {
        btnUndo.isEnabled = currentStep != MeasureStep.BASE
        btnPlace.text = "Зафиксировать"

        when (currentStep) {
            MeasureStep.BASE -> {
                tvStep.text = "ШАГ 1 ИЗ 5 (ДИСТАНЦИЯ)"
                tvHint.text = "Наведите прицел на основание (корни) дерева"
                tvRealtime.visibility = View.GONE
            }
            MeasureStep.TOP -> {
                tvStep.text = "ШАГ 2 ИЗ 5 (ВЫСОТА)"
                tvHint.text = "Ведите прицел вверх до самой макушки дерева"
                tvStatus.text = "AR плоскость больше не нужна. Используется гироскоп."
            }
            MeasureStep.CROWN_LEFT -> {
                tvStep.text = "ШАГ 3 ИЗ 5 (КРОНА)"
                tvHint.text = "Наведите прицел на КРАЙНИЙ ЛЕВЫЙ край веток (кроны)"
            }
            MeasureStep.CROWN_RIGHT -> {
                tvStep.text = "ШАГ 4 ИЗ 5 (КРОНА)"
                tvHint.text = "Наведите прицел на КРАЙНИЙ ПРАВЫЙ край веток"
            }
            MeasureStep.TRUNK_LEFT -> {
                tvStep.text = "ШАГ 5 ИЗ 5 (СТВОЛ)"
                tvHint.text = "Наведите прицел на ЛЕВЫЙ край ствола (на уровне глаз)"
            }
            MeasureStep.TRUNK_RIGHT -> {
                tvStep.text = "ШАГ 5 ИЗ 5 (СТВОЛ)"
                tvHint.text = "Наведите прицел на ПРАВЫЙ край ствола"
                btnPlace.text = "Показать зону"
            }
            MeasureStep.FALL_ZONE -> {
                tvStep.text = "ИТОГ (ЗОНА ПАДЕНИЯ)"
                tvHint.text = "Оцените радиус поражения. Дерево выделено красным."
                tvStatus.text = "Осмотритесь вокруг. Если всё верно — завершайте."
                btnPlace.text = "Сохранить и выйти"
            }
            MeasureStep.DONE -> {
                tvHint.text = "Обработка..."
            }
        }
    }

    private fun safeDouble(d: Double): Double {
        return if (d.isNaN() || d.isInfinite()) 0.0 else d
    }

    private fun finishMeasure() {
        try {
            val base = baseAnchorNode?.anchor?.pose
            val distance = if (base != null) {
                getHorizontalDistance(arFragment.arSceneView.arFrame!!.camera.pose, base)
            } else 0.0

            val json = JSONObject()
                .put("height_m", safeDouble(finalHeight))
                .put("crown_width_m", safeDouble(finalCrownWidth))
                .put("trunk_diameter_m", safeDouble(finalTrunkDiameter))
                .put("distance_m", safeDouble(distance))
                .put("points_count", 6)
                .toString()

            val data = Intent().putExtra(EXTRA_RESULT_JSON, json)
            setResult(Activity.RESULT_OK, data)
            finish()
        } catch (e: Exception) {
            Log.e("ArMeasureActivity", "Error finishing measure", e)
            tvStatus.text = "Произошла ошибка при сохранении"
        }
    }

    // ================= MATH & SENSORS =================

    private fun getCenterHit(frame: Frame): HitResult? {
        val view = arFragment.arSceneView
        val centerX = view.width / 2f
        val centerY = view.height / 2f

        val hits = frame.hitTest(centerX, centerY)
        for (hit in hits) {
            val trackable = hit.trackable
            if (trackable is Plane && trackable.isPoseInPolygon(hit.hitPose)) return hit
            if (trackable is Point && trackable.orientationMode == Point.OrientationMode.ESTIMATED_SURFACE_NORMAL) return hit
        }
        return null
    }

    private fun getHorizontalDistance(cameraPose: Pose, anchorPose: Pose): Double {
        val dx = cameraPose.tx() - anchorPose.tx()
        val dz = cameraPose.tz() - anchorPose.tz()
        return sqrt((dx * dx + dz * dz).toDouble())
    }

    private fun getPitch(cameraPose: Pose): Double {
        val zAxis = cameraPose.zAxis
        // Ограничиваем от -1.0 до 1.0, чтобы asin не возвращал NaN при погрешностях Float
        val forwardY = (-zAxis[1].toDouble()).coerceIn(-1.0, 1.0)
        return asin(forwardY) 
    }

    private fun getYaw(cameraPose: Pose): Double {
        val zAxis = cameraPose.zAxis
        val forwardX = -zAxis[0].toDouble()
        val forwardZ = -zAxis[2].toDouble()
        return atan2(forwardX, forwardZ) 
    }

    private fun getAngleDiff(yaw1: Double, yaw2: Double): Double {
        var diff = abs(yaw1 - yaw2)
        if (diff > PI) diff = 2 * PI - diff
        return diff
    }

    private fun calculateHeight(cameraPose: Pose, anchorPose: Pose, pitchRad: Double, distance: Double): Double {
        val cameraHeightAboveBase = (cameraPose.ty() - anchorPose.ty()).toDouble()
        val topHeightFromCamera = distance * tan(pitchRad)
        return cameraHeightAboveBase + topHeightFromCamera
    }

    private fun vibrate() {
        try {
            val vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createOneShot(40, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                @Suppress("DEPRECATION")
                vibrator.vibrate(40)
            }
        } catch (e: Exception) {
            // Игнорируем
        }
    }

    // ================= ML (В фоне) =================

    private fun processFrame(frame: Frame) {
        try {
            val image = frame.acquireCameraImage()
            mlExecutor.execute {
                try {
                    val bitmap = imageToBitmap(image)
                    val resized = android.graphics.Bitmap.createScaledBitmap(bitmap, 320, 320, true)
                    treeDetector.detect(resized)
                } catch (e: Exception) {
                    // ignore
                } finally {
                    image.close()
                }
            }
        } catch (e: Exception) {
            // ignore
        }
    }

    private fun imageToBitmap(image: android.media.Image): android.graphics.Bitmap {
        val plane = image.planes[0]
        val buffer = plane.buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            ?: android.graphics.Bitmap.createBitmap(32, 32, android.graphics.Bitmap.Config.ARGB_8888)
    }

    override fun onDestroy() {
        try {
            baseAnchorNode?.anchor?.detach()
            fallZoneNode?.setParent(null)
            mlExecutor.shutdownNow()
        } catch (e: Exception) {}
        super.onDestroy()
    }
}