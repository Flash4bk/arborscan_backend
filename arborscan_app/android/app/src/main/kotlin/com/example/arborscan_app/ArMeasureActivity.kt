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
import com.google.ar.core.Frame
import com.google.ar.core.HitResult
import com.google.ar.core.Plane
import com.google.ar.core.Point
import com.google.ar.core.Pose
import com.google.ar.core.TrackingState
import com.google.ar.sceneform.AnchorNode
import com.google.ar.sceneform.FrameTime
import com.google.ar.sceneform.math.Vector3
import com.google.ar.sceneform.rendering.Color as SceneColor
import com.google.ar.sceneform.rendering.MaterialFactory
import com.google.ar.sceneform.rendering.ShapeFactory
import com.google.ar.sceneform.ux.ArFragment
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs

/**
 * ArborScan AR alpha 2.
 *
 * Changes from alpha 1:
 *  - geometric Measurement Coach before height and DBH;
 *  - height uses a fixed-plane ray method; field accuracy is unvalidated;
 *  - crown is not measured in AR; no scale is transferred to photo widths
 *    without a separate image calibration;
 *  - DBH uses cylindrical tangent-ray geometry, 3 repeats and median;
 *  - movement between left/right DBH edges is quality-gated;
 *  - user-facing "94% quality" is removed. We expose diagnostic statuses
 *    instead; the numeric quality field remains internal for backend fusion.
 */
class ArMeasureActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_RESULT_JSON = "result_json"

        private const val DBH_HEIGHT_M = 1.30
        private const val DBH_HEIGHT_TOLERANCE_M = 0.12
        private const val MIN_INCIDENCE = 0.18

        private const val HEIGHT_POSITION_MIN_M = 3.0
        private const val HEIGHT_POSITION_MAX_M = 15.0
        private const val HEIGHT_POSITION_IDEAL_MIN_M = 4.0
        private const val HEIGHT_POSITION_IDEAL_MAX_M = 10.0

        private const val DBH_POSITION_MIN_M = 2.0
        private const val DBH_POSITION_MAX_M = 5.5
        private const val DBH_POSITION_IDEAL_MIN_M = 2.5
        private const val DBH_POSITION_IDEAL_MAX_M = 4.5

        private const val MAX_DBH_PAIR_TRANSLATION_M = 0.08
        private const val REQUIRED_DBH_REPEATS = 3
        private const val MAX_DBH_REL_SPREAD = 0.12
        private const val MAX_DBH_ABS_SPREAD_M = 0.030
        private const val STABLE_CAMERA_SPEED_M_S = 0.22
    }

    enum class MeasureStep {
        BASE,
        HEIGHT_POSITION,
        TOP,
        DBH_POSITION,
        DBH_LEFT,
        DBH_RIGHT,
        REVIEW,
        DONE,
    }

    data class Observation(
        val camera: ArGeometry.Vec3,
        val forward: ArGeometry.Vec3,
        val point: ArGeometry.Vec3,
        val incidence: Double,
        val distanceAlongRayM: Double,
        val timestampMs: Long,
    )

    data class PositionAssessment(
        val distanceM: Double,
        val viewIncidence: Double,
        val cameraSpeedMps: Double,
        val score: Double,
        val status: String,
        val hint: String,
    )

    data class DbhSample(
        val left: Observation,
        val right: Observation,
        val diameterM: Double,
        val measurementHeightM: Double,
        val cameraDistanceM: Double,
        val angularSpanRad: Double,
        val pairTranslationM: Double,
    )

    private lateinit var arFragment: ArFragment
    private lateinit var tvStep: TextView
    private lateinit var tvHint: TextView
    private lateinit var tvStatus: TextView
    private lateinit var tvRealtime: TextView
    private lateinit var btnPlace: Button
    private lateinit var btnUndo: ImageButton

    private var currentStep = MeasureStep.BASE
    private var baseAnchorNode: AnchorNode? = null
    private var basePoint: ArGeometry.Vec3? = null
    private var treePlane: ArGeometry.Plane? = null
    private var baseSurfaceKind = "unknown"
    private var baseFixCamera: ArGeometry.Vec3? = null
    private var baseDistanceM: Double? = null

    private var heightPositionAssessment: PositionAssessment? = null
    private var dbhPositionAssessment: PositionAssessment? = null
    private var topObservation: Observation? = null
    private var dbhLeftObservation: Observation? = null
    private val dbhSamples = mutableListOf<DbhSample>()

    private var finalHeightM: Double? = null
    private var finalTrunkDiameterM: Double? = null
    private var dbhMeasuredHeightM: Double? = null
    private var dbhRepeatSpreadM: Double? = null
    private var dbhRepeatSpreadRatio: Double? = null
    private var lastRepeatFailure: String? = null

    private var totalFrames = 0L
    private var trackingFrames = 0L
    private var trackingLossEvents = 0
    private var wasTracking = false
    private var consecutiveTrackingFrames = 0

    private var lastCameraPosition: ArGeometry.Vec3? = null
    private var lastCameraTimestampNs: Long? = null
    private var smoothedCameraSpeedMps = 0.0

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
        arFragment.setOnTapArPlaneListener { _, _, _ -> }

        btnPlace.setOnClickListener { onPlaceClicked() }
        btnUndo.setOnClickListener { undoStep() }

        updateUi()
    }

    private fun onSceneUpdate(@Suppress("UNUSED_PARAMETER") frameTime: FrameTime) {
        val frame = arFragment.arSceneView.arFrame ?: return
        totalFrames += 1

        val tracking = frame.camera.trackingState == TrackingState.TRACKING
        if (tracking) {
            trackingFrames += 1
            consecutiveTrackingFrames += 1
            updateCameraMotion(frame.camera.pose)
        } else {
            consecutiveTrackingFrames = 0
            if (wasTracking) trackingLossEvents += 1
        }
        wasTracking = tracking

        if (!tracking) {
            if (basePoint != null) {
                clearBase()
                currentStep = MeasureStep.BASE
                updateUi()
            }
            btnPlace.isEnabled = false
            tvStatus.text = "AR-трекинг потерян. Двигайте телефон медленно и дайте ARCore восстановиться."
            return
        }
        if (baseAnchorNode?.anchor?.trackingState?.let { it != TrackingState.TRACKING } == true) {
            clearBase()
            currentStep = MeasureStep.BASE
            updateUi()
            tvStatus.text = "Привязка основания потеряна. Повторно отметьте землю у дерева."
            return
        }

        when (currentStep) {
            MeasureStep.BASE -> updateBaseStep(frame)
            MeasureStep.HEIGHT_POSITION -> updateCoachStep(frame.camera.pose, forDbh = false)
            MeasureStep.TOP -> updateTopStep(frame.camera.pose)
            MeasureStep.DBH_POSITION -> updateCoachStep(frame.camera.pose, forDbh = true)
            MeasureStep.DBH_LEFT -> updateDbhLeftStep(frame.camera.pose)
            MeasureStep.DBH_RIGHT -> updateDbhRightStep(frame.camera.pose)
            MeasureStep.REVIEW -> {
                btnPlace.isEnabled = true
                updateReviewText()
            }
            MeasureStep.DONE -> btnPlace.isEnabled = false
        }
    }

    private fun updateCameraMotion(cameraPose: Pose) {
        val now = System.nanoTime()
        val current = posePosition(cameraPose)
        val previous = lastCameraPosition
        val previousTs = lastCameraTimestampNs

        if (previous != null && previousTs != null) {
            val dt = (now - previousTs) / 1_000_000_000.0
            if (dt in 0.005..0.25) {
                val instant = ArGeometry.horizontalDistance(current, previous) / dt
                if (instant.isFinite()) {
                    smoothedCameraSpeedMps = 0.82 * smoothedCameraSpeedMps + 0.18 * instant.coerceAtMost(4.0)
                }
            }
        }

        lastCameraPosition = current
        lastCameraTimestampNs = now
    }

    private fun updateBaseStep(frame: Frame) {
        val hit = getCenterHit(frame)
        btnPlace.isEnabled = hit != null && consecutiveTrackingFrames >= 10
        tvRealtime.visibility = View.GONE
        tvStatus.text = when {
            hit == null -> "Помощник: найдите поверхность у основания ствола. Медленно поводите камерой."
            consecutiveTrackingFrames < 10 -> "Помощник: поверхность найдена, подождите секунду для стабилизации AR."
            else -> "Помощник: основание видно. Наведите прицел на место выхода ствола из земли."
        }
    }

    private fun updateCoachStep(cameraPose: Pose, forDbh: Boolean) {
        val assessment = assessPosition(cameraPose, forDbh)
        if (assessment == null) {
            btnPlace.isEnabled = false
            tvRealtime.visibility = View.GONE
            tvStatus.text = "Помощник не может оценить позицию. Вернитесь к дереву и наведите телефон на ствол."
            return
        }

        tvRealtime.visibility = View.VISIBLE
        val title = when (assessment.status) {
            "good" -> "Позиция подходит"
            "usable" -> "Позиция допустима"
            else -> "Измените позицию"
        }
        tvRealtime.text = "$title\nДистанция: %.1f м".format(assessment.distanceM)
        tvStatus.text = assessment.hint
        btnPlace.isEnabled = assessment.score >= 0.58 && consecutiveTrackingFrames >= 15
    }

    private fun assessPosition(cameraPose: Pose, forDbh: Boolean): PositionAssessment? {
        val base = basePoint ?: return null
        val plane = treePlane ?: return null
        val camera = posePosition(cameraPose)
        val distance = ArGeometry.horizontalDistance(camera, base)
        if (!distance.isFinite()) return null

        val towardBase = ArGeometry.Vec3(base.x - camera.x, 0.0, base.z - camera.z).normalized() ?: return null
        val planeNormal = plane.normal.normalized() ?: return null
        val incidence = abs(towardBase.dot(planeNormal)).coerceIn(0.0, 1.0)
        val speed = smoothedCameraSpeedMps

        val min = if (forDbh) DBH_POSITION_MIN_M else HEIGHT_POSITION_MIN_M
        val max = if (forDbh) DBH_POSITION_MAX_M else HEIGHT_POSITION_MAX_M
        val idealMin = if (forDbh) DBH_POSITION_IDEAL_MIN_M else HEIGHT_POSITION_IDEAL_MIN_M
        val idealMax = if (forDbh) DBH_POSITION_IDEAL_MAX_M else HEIGHT_POSITION_IDEAL_MAX_M

        val distanceScore = when {
            distance < min -> (distance / min).coerceIn(0.0, 1.0) * 0.55
            distance > max -> (max / distance).coerceIn(0.0, 1.0) * 0.65
            distance in idealMin..idealMax -> 1.0
            else -> 0.78
        }
        val viewScore = ((incidence - 0.25) / 0.65).coerceIn(0.0, 1.0)
        val motionScore = when {
            speed <= 0.08 -> 1.0
            speed <= STABLE_CAMERA_SPEED_M_S -> 0.82
            speed <= 0.45 -> 0.55
            else -> 0.20
        }
        val trackingScore = if (consecutiveTrackingFrames >= 30) 1.0 else 0.75
        val surfaceScore = when (baseSurfaceKind) {
            "plane" -> 1.0
            "feature_point" -> 0.78
            else -> 0.65
        }

        val score = (
            0.34 * distanceScore +
                0.24 * viewScore +
                0.22 * motionScore +
                0.12 * trackingScore +
                0.08 * surfaceScore
            ).coerceIn(0.0, 1.0)

        val hint = when {
            distance < min -> "Помощник: слишком близко. Отойдите от дерева примерно на %.1f м.".format(min - distance + 0.5)
            distance > max -> "Помощник: слишком далеко. Подойдите ближе примерно на %.1f м.".format(distance - max + 0.5)
            incidence < 0.45 -> "Помощник: ракурс слишком боковой. Сместитесь так, чтобы смотреть на ствол более прямо."
            speed > STABLE_CAMERA_SPEED_M_S -> "Помощник: остановитесь и удерживайте телефон спокойнее."
            distance !in idealMin..idealMax -> if (forDbh) {
                "Помощник: можно измерять, но для диаметра лучше дистанция %.1f–%.1f м.".format(idealMin, idealMax)
            } else {
                "Помощник: позиция допустима. Для высоты обычно лучше %.1f–%.1f м.".format(idealMin, idealMax)
            }
            else -> if (forDbh) {
                "Помощник: позиция для диаметра хорошая. После фиксации левого края не делайте шагов."
            } else {
                "Помощник: позиция хорошая. Для верхушки можно немного повернуть или наклонить телефон без смены точки."
            }
        }

        val status = when {
            score >= 0.78 -> "good"
            score >= 0.58 -> "usable"
            else -> "poor"
        }
        return PositionAssessment(distance, incidence, speed, score, status, hint)
    }

    private fun currentObservation(cameraPose: Pose): Observation? {
        val plane = treePlane ?: return null
        val camera = posePosition(cameraPose)
        val forward = poseForward(cameraPose)
        val intersection = ArGeometry.intersect(ArGeometry.Ray(camera, forward), plane) ?: return null
        return Observation(
            camera = camera,
            forward = forward,
            point = intersection.point,
            incidence = intersection.incidence,
            distanceAlongRayM = intersection.distanceAlongRayM,
            timestampMs = System.currentTimeMillis(),
        )
    }

    private fun updateTopStep(cameraPose: Pose) {
        val base = basePoint
        val observation = currentObservation(cameraPose)
        tvRealtime.visibility = View.VISIBLE

        if (base == null || observation == null) {
            btnPlace.isEnabled = false
            tvRealtime.text = "Нет геометрии"
            tvStatus.text = "Помощник: вернитесь к дереву и наведите прицел на верхушку."
            return
        }

        val height = ArGeometry.heightAboveBase(base, observation.point)
        val incidenceOk = observation.incidence >= MIN_INCIDENCE
        btnPlace.isEnabled = incidenceOk && height > 0.5
        tvRealtime.text = "Высота: %.2f м".format(height)
        tvStatus.text = when {
            observation.incidence < 0.35 -> "Помощник: слишком косой ракурс. Сместитесь ближе к направлению на ствол."
            smoothedCameraSpeedMps > 0.45 -> "Помощник: удерживайте телефон спокойнее перед фиксацией верхушки."
            else -> "Помощник: наведите точно на самую верхнюю живую точку дерева."
        }
    }

    private fun updateDbhLeftStep(cameraPose: Pose) {
        val base = basePoint ?: return
        val observation = currentObservation(cameraPose)
        tvRealtime.visibility = View.VISIBLE

        if (observation == null) {
            btnPlace.isEnabled = false
            tvRealtime.text = "Диаметр ${dbhSamples.size + 1}/$REQUIRED_DBH_REPEATS"
            tvStatus.text = "Помощник: прицел не пересекает измерительную область."
            return
        }

        val h = ArGeometry.heightAboveBase(base, observation.point)
        val hOk = ArGeometry.isDbhHeight(h, DBH_HEIGHT_M, DBH_HEIGHT_TOLERANCE_M)
        val position = assessPosition(cameraPose, forDbh = true)
        val positionOk = position != null && position.score >= 0.58

        btnPlace.isEnabled = hOk && observation.incidence >= MIN_INCIDENCE && positionOk
        tvRealtime.text = "Диаметр ${dbhSamples.size + 1}/$REQUIRED_DBH_REPEATS\nВысота точки: %.2f м".format(h)
        tvStatus.text = when {
            !positionOk -> position?.hint ?: "Помощник: выберите стабильную позицию для диаметра."
            !hOk -> "Помощник: наведите левый край ствола на высоте 1,30 м (±12 см)."
            smoothedCameraSpeedMps > STABLE_CAMERA_SPEED_M_S -> "Помощник: остановитесь. После левого края нельзя делать шаг до правого края."
            else -> "Помощник: фиксируйте левый край. Затем только поверните телефон к правому краю, не меняя место."
        }
    }

    private fun updateDbhRightStep(cameraPose: Pose) {
        val base = basePoint ?: return
        val left = dbhLeftObservation
        val right = currentObservation(cameraPose)
        tvRealtime.visibility = View.VISIBLE

        if (left == null || right == null) {
            btnPlace.isEnabled = false
            tvRealtime.text = "Диаметр ${dbhSamples.size + 1}/$REQUIRED_DBH_REPEATS"
            tvStatus.text = "Помощник: потеряна геометрия правого края."
            return
        }

        val h = ArGeometry.heightAboveBase(base, right.point)
        val hOk = ArGeometry.isDbhHeight(h, DBH_HEIGHT_M, DBH_HEIGHT_TOLERANCE_M)
        val estimate = ArGeometry.cylinderDiameterFromTangentRays(
            base,
            left.camera,
            left.forward,
            right.camera,
            right.forward,
        )
        val translation = ArGeometry.horizontalDistance(left.camera, right.camera)
        val moveOk = translation <= MAX_DBH_PAIR_TRANSLATION_M
        val diameter = estimate?.diameterM
        val diameterOk = diameter != null && diameter in 0.03..3.0

        btnPlace.isEnabled = hOk && moveOk && diameterOk && right.incidence >= MIN_INCIDENCE
        tvRealtime.text = if (diameter != null) {
            "Диаметр ${dbhSamples.size + 1}/$REQUIRED_DBH_REPEATS: %.3f м\nСмещение: %.0f мм".format(diameter, translation * 1000.0)
        } else {
            "Диаметр ${dbhSamples.size + 1}/$REQUIRED_DBH_REPEATS"
        }
        tvStatus.text = when {
            !moveOk -> "Помощник: телефон сместился на %.0f мм. Вернитесь к левому краю и повторите пару без шага.".format(translation * 1000.0)
            !hOk -> "Помощник: правый край тоже должен быть на высоте 1,30 м."
            !diameterOk -> "Помощник: геометрия ствола неустойчива. Повторите левый край."
            else -> "Помощник: положение сохранено. Фиксируйте правый край."
        }
    }

    private fun onPlaceClicked() {
        try {
            vibrate()
            val frame = arFragment.arSceneView.arFrame ?: return
            if (frame.camera.trackingState != TrackingState.TRACKING) return

            when (currentStep) {
                MeasureStep.BASE -> fixBase(frame)

                MeasureStep.HEIGHT_POSITION -> {
                    val assessment = assessPosition(frame.camera.pose, forDbh = false) ?: return
                    if (assessment.score < 0.58) return
                    heightPositionAssessment = assessment
                    currentStep = MeasureStep.TOP
                }

                MeasureStep.TOP -> {
                    val base = basePoint ?: return
                    val obs = currentObservation(frame.camera.pose) ?: return
                    val h = ArGeometry.heightAboveBase(base, obs.point)
                    if (obs.incidence < MIN_INCIDENCE || h <= 0.5) return
                    topObservation = obs
                    finalHeightM = h
                    currentStep = MeasureStep.DBH_POSITION
                }

                MeasureStep.DBH_POSITION -> {
                    val assessment = assessPosition(frame.camera.pose, forDbh = true) ?: return
                    if (assessment.score < 0.58) return
                    dbhPositionAssessment = assessment
                    resetDbhSeries()
                    currentStep = MeasureStep.DBH_LEFT
                }

                MeasureStep.DBH_LEFT -> {
                    val base = basePoint ?: return
                    val obs = currentObservation(frame.camera.pose) ?: return
                    val h = ArGeometry.heightAboveBase(base, obs.point)
                    if (!ArGeometry.isDbhHeight(h, DBH_HEIGHT_M, DBH_HEIGHT_TOLERANCE_M)) return
                    val assessment = assessPosition(frame.camera.pose, forDbh = true) ?: return
                    if (assessment.score < 0.58) return
                    dbhLeftObservation = obs
                    currentStep = MeasureStep.DBH_RIGHT
                }

                MeasureStep.DBH_RIGHT -> acceptDbhRight(frame.camera.pose)

                MeasureStep.REVIEW -> {
                    currentStep = MeasureStep.DONE
                    finishMeasure()
                    return
                }

                MeasureStep.DONE -> return
            }
            updateUi()
        } catch (e: Exception) {
            Log.e("ArMeasureActivity", "Measurement step failed", e)
            tvStatus.text = "Не удалось зафиксировать измерение. Повторите наведение."
        }
    }

    private fun fixBase(frame: Frame) {
        val hit = getCenterHit(frame) ?: return
        val cameraPose = frame.camera.pose
        val base = posePosition(hit.hitPose)
        val camera = posePosition(cameraPose)
        val distance = ArGeometry.horizontalDistance(camera, base)
        if (!distance.isFinite() || distance < 0.7) {
            tvStatus.text = "Помощник: слишком близко к основанию. Отойдите хотя бы на 1–2 м и повторите."
            return
        }

        val plane = ArGeometry.buildVerticalTreePlane(base, camera)
        if (plane == null) {
            tvStatus.text = "Помощник: не удалось построить геометрию. Измените позицию."
            return
        }

        basePoint = base
        baseFixCamera = camera
        baseDistanceM = distance
        treePlane = plane
        baseSurfaceKind = when (hit.trackable) {
            is Plane -> "plane"
            is Point -> "feature_point"
            else -> "other"
        }

        val anchor = hit.createAnchor()
        baseAnchorNode?.anchor?.detach()
        baseAnchorNode?.setParent(null)
        baseAnchorNode = AnchorNode(anchor).apply {
            setParent(arFragment.arSceneView.scene)
        }

        MaterialFactory.makeOpaqueWithColor(this, SceneColor(0.27f, 0.88f, 0.63f)).thenAccept { material ->
            baseAnchorNode?.renderable = ShapeFactory.makeSphere(0.05f, Vector3.zero(), material)
        }

        currentStep = MeasureStep.HEIGHT_POSITION
    }

    private fun acceptDbhRight(cameraPose: Pose) {
        val base = basePoint ?: return
        val left = dbhLeftObservation ?: return
        val right = currentObservation(cameraPose) ?: return
        val hRight = ArGeometry.heightAboveBase(base, right.point)
        if (!ArGeometry.isDbhHeight(hRight, DBH_HEIGHT_M, DBH_HEIGHT_TOLERANCE_M)) return

        val estimate = ArGeometry.cylinderDiameterFromTangentRays(
            base,
            left.camera,
            left.forward,
            right.camera,
            right.forward,
        ) ?: return

        if (estimate.cameraTranslationM > MAX_DBH_PAIR_TRANSLATION_M) {
            tvStatus.text = "Телефон сместился слишком сильно. Повторите левый/правый край без шага."
            dbhLeftObservation = null
            currentStep = MeasureStep.DBH_LEFT
            return
        }
        if (estimate.diameterM !in 0.03..3.0) {
            dbhLeftObservation = null
            currentStep = MeasureStep.DBH_LEFT
            return
        }

        val hLeft = ArGeometry.heightAboveBase(base, left.point)
        dbhSamples += DbhSample(
            left = left,
            right = right,
            diameterM = estimate.diameterM,
            measurementHeightM = (hLeft + hRight) / 2.0,
            cameraDistanceM = estimate.cameraToAxisM,
            angularSpanRad = estimate.angularSpanRad,
            pairTranslationM = estimate.cameraTranslationM,
        )
        dbhLeftObservation = null

        if (dbhSamples.size < REQUIRED_DBH_REPEATS) {
            currentStep = MeasureStep.DBH_LEFT
            return
        }

        val diameters = dbhSamples.map { it.diameterM }
        val median = ArGeometry.median(diameters) ?: return
        val spread = diameters.maxOrNull()!! - diameters.minOrNull()!!
        val relSpread = ArGeometry.relativeSpread(diameters) ?: 0.0

        if (spread > MAX_DBH_ABS_SPREAD_M || relSpread > MAX_DBH_REL_SPREAD) {
            lastRepeatFailure = "Разброс диаметра: %.0f мм (%.1f%%). Сделайте три повтора ещё раз.".format(spread * 1000.0, relSpread * 100.0)
            resetDbhSeries(keepFailure = true)
            currentStep = MeasureStep.DBH_LEFT
            tvStatus.text = lastRepeatFailure
            return
        }

        finalTrunkDiameterM = median
        dbhMeasuredHeightM = ArGeometry.median(dbhSamples.map { it.measurementHeightM })
        dbhRepeatSpreadM = spread
        dbhRepeatSpreadRatio = relSpread
        currentStep = MeasureStep.REVIEW
    }

    private fun resetDbhSeries(keepFailure: Boolean = false) {
        dbhSamples.clear()
        dbhLeftObservation = null
        finalTrunkDiameterM = null
        dbhMeasuredHeightM = null
        dbhRepeatSpreadM = null
        dbhRepeatSpreadRatio = null
        if (!keepFailure) lastRepeatFailure = null
    }

    private fun undoStep() {
        try {
            vibrate()
            when (currentStep) {
                MeasureStep.BASE -> Unit
                MeasureStep.HEIGHT_POSITION -> {
                    clearBase()
                    currentStep = MeasureStep.BASE
                }
                MeasureStep.TOP -> currentStep = MeasureStep.HEIGHT_POSITION
                MeasureStep.DBH_POSITION -> {
                    topObservation = null
                    finalHeightM = null
                    currentStep = MeasureStep.TOP
                }
                MeasureStep.DBH_LEFT -> {
                    if (dbhSamples.isNotEmpty()) dbhSamples.removeAt(dbhSamples.lastIndex)
                    else currentStep = MeasureStep.DBH_POSITION
                }
                MeasureStep.DBH_RIGHT -> {
                    dbhLeftObservation = null
                    currentStep = MeasureStep.DBH_LEFT
                }
                MeasureStep.REVIEW -> {
                    resetDbhSeries()
                    currentStep = MeasureStep.DBH_LEFT
                }
                MeasureStep.DONE -> Unit
            }
            updateUi()
        } catch (e: Exception) {
            Log.e("ArMeasureActivity", "Undo failed", e)
        }
    }

    private fun clearBase() {
        baseAnchorNode?.anchor?.detach()
        baseAnchorNode?.setParent(null)
        baseAnchorNode = null
        basePoint = null
        treePlane = null
        baseFixCamera = null
        baseDistanceM = null
        baseSurfaceKind = "unknown"
        heightPositionAssessment = null
        dbhPositionAssessment = null
        topObservation = null
        finalHeightM = null
        resetDbhSeries()
    }

    private fun updateUi() {
        btnUndo.isEnabled = currentStep != MeasureStep.BASE && currentStep != MeasureStep.DONE
        btnPlace.text = "Зафиксировать"

        when (currentStep) {
            MeasureStep.BASE -> {
                tvStep.text = "1 / 5 • ОСНОВАНИЕ"
                tvHint.text = "Зафиксируйте основание ствола"
                tvRealtime.visibility = View.GONE
            }
            MeasureStep.HEIGHT_POSITION -> {
                tvStep.text = "2 / 5 • ПОЗИЦИЯ"
                tvHint.text = "Помощник подбирает место для измерения высоты"
                btnPlace.text = "Позиция подходит"
            }
            MeasureStep.TOP -> {
                tvStep.text = "3 / 5 • ВЫСОТА"
                tvHint.text = "Наведите на самую верхнюю точку дерева"
            }
            MeasureStep.DBH_POSITION -> {
                tvStep.text = "4 / 5 • ПОЗИЦИЯ ДЛЯ ДИАМЕТРА"
                tvHint.text = "Подберите удобную дистанцию к стволу"
                btnPlace.text = "Измерить диаметр"
            }
            MeasureStep.DBH_LEFT -> {
                tvStep.text = "5 / 5 • ДИАМЕТР ${dbhSamples.size + 1}/$REQUIRED_DBH_REPEATS"
                tvHint.text = "Левый край ствола на высоте 1,30 м"
                btnPlace.text = "Левый край"
                if (lastRepeatFailure != null) tvStatus.text = lastRepeatFailure
            }
            MeasureStep.DBH_RIGHT -> {
                tvStep.text = "5 / 5 • ДИАМЕТР ${dbhSamples.size + 1}/$REQUIRED_DBH_REPEATS"
                tvHint.text = "Правый край — не меняйте положение телефона"
                btnPlace.text = "Правый край"
            }
            MeasureStep.REVIEW -> {
                tvStep.text = "ПРОВЕРКА ИЗМЕРЕНИЙ"
                tvHint.text = "Ширина кроны требует отдельного измерения на фото"
                btnPlace.text = "Сохранить"
                updateReviewText()
            }
            MeasureStep.DONE -> {
                tvStep.text = "ГОТОВО"
                tvHint.text = "Сохранение измерений..."
                btnPlace.isEnabled = false
            }
        }
    }

    private fun updateReviewText() {
        val diagnostics = computeDiagnostics()
        tvRealtime.visibility = View.VISIBLE
        tvRealtime.text = buildString {
            append("Высота  %.2f м\n".format(finalHeightM ?: 0.0))
            append("Диаметр  %.3f м на высоте %.2f м\n".format(finalTrunkDiameterM ?: 0.0, dbhMeasuredHeightM ?: 0.0))
            append("Крона    не измерена\n")
            append("Разброс диаметра  %.0f мм".format((dbhRepeatSpreadM ?: 0.0) * 1000.0))
        }
        tvStatus.text = buildString {
            append("AR-трекинг: ${diagnostics.getString("tracking_status_ru")}. ")
            append("Геометрия: ${diagnostics.getString("geometry_status_ru")}. ")
            append("Повторяемость диаметра: ${diagnostics.getString("dbh_repeatability_status_ru")}.")
        }
    }

    private fun finishMeasure() {
        val height = finalHeightM
        val trunk = finalTrunkDiameterM
        val dbhHeight = dbhMeasuredHeightM
        if (height == null || trunk == null || dbhHeight == null || dbhSamples.size != REQUIRED_DBH_REPEATS) {
            currentStep = MeasureStep.REVIEW
            tvStatus.text = "Измерения неполные. Повторите диаметр."
            return
        }

        val diagnostics = computeDiagnostics()
        val warnings = diagnostics.getJSONArray("warnings")
        val internalQuality = diagnostics.getDouble("internal_quality")

        val json = JSONObject()
            .put("schema_version", "ar_measurement_v4")
            .put("session_id", java.util.UUID.randomUUID().toString())
            .put("captured_at_ms", System.currentTimeMillis())
            .put("coordinate_system", "arcore_world_m_y_up")
            .put("quality_interpretation", "engineering_heuristic_not_accuracy")
            .put("height_m", finiteOrNull(height))
            .put("crown_width_m", JSONObject.NULL)
            .put("trunk_diameter_m", finiteOrNull(trunk))
            .put("trunk_measurement_height_m", finiteOrNull(dbhHeight))
            .put("distance_m", finiteOrNull(baseDistanceM))
            .put("quality", internalQuality)
            .put("overall_status", diagnostics.getString("overall_status"))
            .put("points_count", 2 + REQUIRED_DBH_REPEATS * 2)
            .put("base_surface", baseSurfaceKind)
            .put("tracking_ratio", trackingRatio())
            .put("tracking_loss_events", trackingLossEvents)
            .put("warnings", warnings)
            .put("coach", coachAuditJson())
            .put("dbh_repeat_spread_m", finiteOrNull(dbhRepeatSpreadM))
            .put("dbh_repeat_spread_ratio", finiteOrNull(dbhRepeatSpreadRatio))
            .put("geometry", geometryAuditJson())
            .toString()

        val data = Intent().putExtra(EXTRA_RESULT_JSON, json)
        setResult(Activity.RESULT_OK, data)
        finish()
    }

    private fun computeDiagnostics(): JSONObject {
        val warnings = mutableListOf<String>()
        val tracking = trackingRatio()
        val trackingStatus = when {
            tracking >= 0.95 && trackingLossEvents <= 1 -> "excellent"
            tracking >= 0.85 -> "good"
            else -> "weak"
        }
        if (trackingStatus == "weak") warnings += "ar_tracking_unstable"

        val heightPos = heightPositionAssessment?.status ?: "poor"
        val dbhPos = dbhPositionAssessment?.status ?: "poor"
        val topIncidence = topObservation?.incidence ?: 0.0
        val geometryStatus = when {
            heightPos == "good" && dbhPos == "good" && topIncidence >= 0.45 -> "good"
            heightPos != "poor" && dbhPos != "poor" && topIncidence >= MIN_INCIDENCE -> "usable"
            else -> "weak"
        }
        if (geometryStatus == "weak") warnings += "ar_geometry_weak"

        val relSpread = dbhRepeatSpreadRatio ?: 1.0
        val absSpread = dbhRepeatSpreadM ?: 999.0
        val dbhRepeatStatus = when {
            relSpread <= 0.06 && absSpread <= 0.015 -> "excellent"
            relSpread <= MAX_DBH_REL_SPREAD && absSpread <= MAX_DBH_ABS_SPREAD_M -> "good"
            else -> "repeat"
        }
        if (dbhRepeatStatus == "repeat") warnings += "dbh_repeatability_failed"

        val overallStatus = when {
            trackingStatus == "excellent" && geometryStatus == "good" && dbhRepeatStatus != "repeat" -> "good"
            trackingStatus != "weak" && geometryStatus != "weak" && dbhRepeatStatus != "repeat" -> "usable"
            else -> "repeat_recommended"
        }

        val trackingScore = when (trackingStatus) { "excellent" -> 1.0; "good" -> 0.82; else -> 0.55 }
        val geometryScore = when (geometryStatus) { "good" -> 1.0; "usable" -> 0.78; else -> 0.45 }
        val repeatScore = when (dbhRepeatStatus) { "excellent" -> 1.0; "good" -> 0.84; else -> 0.35 }
        val internalQuality = (0.40 * trackingScore + 0.35 * geometryScore + 0.25 * repeatScore)
            .coerceIn(0.0, 0.85)

        return JSONObject()
            .put("overall_status", overallStatus)
            .put("tracking_status", trackingStatus)
            .put("geometry_status", geometryStatus)
            .put("dbh_repeatability_status", dbhRepeatStatus)
            .put("tracking_status_ru", when (trackingStatus) {
                "excellent" -> "отличный"
                "good" -> "хороший"
                else -> "слабый"
            })
            .put("geometry_status_ru", when (geometryStatus) {
                "good" -> "хорошая"
                "usable" -> "допустимая"
                else -> "слабая"
            })
            .put("dbh_repeatability_status_ru", when (dbhRepeatStatus) {
                "excellent" -> "отличная"
                "good" -> "хорошая"
                else -> "нужно повторить"
            })
            .put("internal_quality", internalQuality)
            .put("warnings", JSONArray(warnings.distinct()))
    }


    private fun trackingRatio(): Double {
        if (totalFrames <= 0L) return 0.0
        return (trackingFrames.toDouble() / totalFrames.toDouble()).coerceIn(0.0, 1.0)
    }

    private fun coachAuditJson(): JSONObject {
        fun assessment(a: PositionAssessment?): Any {
            if (a == null) return JSONObject.NULL
            return JSONObject()
                .put("distance_m", a.distanceM)
                .put("view_incidence", a.viewIncidence)
                .put("camera_speed_m_s", a.cameraSpeedMps)
                .put("score", a.score)
                .put("status", a.status)
        }
        return JSONObject()
            .put("height_position", assessment(heightPositionAssessment))
            .put("dbh_position", assessment(dbhPositionAssessment))
            .put("height_distance_range_m", JSONArray(listOf(HEIGHT_POSITION_MIN_M, HEIGHT_POSITION_MAX_M)))
            .put("dbh_distance_range_m", JSONArray(listOf(DBH_POSITION_MIN_M, DBH_POSITION_MAX_M)))
            .put("dbh_pair_max_translation_m", MAX_DBH_PAIR_TRANSLATION_M)
    }

    private fun geometryAuditJson(): JSONObject {
        fun vec(v: ArGeometry.Vec3?): Any {
            if (v == null) return JSONObject.NULL
            return JSONObject().put("x", v.x).put("y", v.y).put("z", v.z)
        }

        fun obs(o: Observation?): Any {
            if (o == null) return JSONObject.NULL
            return JSONObject()
                .put("camera", vec(o.camera))
                .put("forward", vec(o.forward))
                .put("intersection", vec(o.point))
                .put("incidence", o.incidence)
                .put("ray_distance_m", o.distanceAlongRayM)
                .put("timestamp_ms", o.timestampMs)
        }

        val samplesJson = JSONArray()
        dbhSamples.forEachIndexed { index, sample ->
            samplesJson.put(
                JSONObject()
                    .put("repeat", index + 1)
                    .put("diameter_m", sample.diameterM)
                    .put("measurement_height_m", sample.measurementHeightM)
                    .put("camera_distance_m", sample.cameraDistanceM)
                    .put("angular_span_rad", sample.angularSpanRad)
                    .put("pair_translation_m", sample.pairTranslationM)
                    .put("left", obs(sample.left))
                    .put("right", obs(sample.right))
            )
        }

        val plane = treePlane
        return JSONObject()
            .put("height_method", "fixed_vertical_tree_plane_ray_intersection_v1")
            .put("dbh_method", "cylindrical_tangent_rays_median_v2")
            .put("dbh_protocol_verified", false)
            .put("diameter_limitations", "vertical_cylinder_assumption; field_base_slope_fork_lean_rules_not_verified")
            .put("crown_method", "not_measured_requires_explicit_photo_calibration")
            .put("base", vec(basePoint))
            .put("base_camera", vec(baseFixCamera))
            .put("plane_normal", vec(plane?.normal))
            .put("top", obs(topObservation))
            .put("dbh_samples", samplesJson)
    }

    private fun getCenterHit(frame: Frame): HitResult? {
        val view = arFragment.arSceneView
        val centerX = view.width / 2f
        val centerY = view.height / 2f
        for (hit in frame.hitTest(centerX, centerY)) {
            val trackable = hit.trackable
            if (trackable is Plane && trackable.type == Plane.Type.HORIZONTAL_UPWARD_FACING &&
                trackable.trackingState == TrackingState.TRACKING &&
                trackable.isPoseInPolygon(hit.hitPose)) return hit
        }
        return null
    }

    private fun posePosition(pose: Pose): ArGeometry.Vec3 = ArGeometry.Vec3(
        pose.tx().toDouble(),
        pose.ty().toDouble(),
        pose.tz().toDouble(),
    )

    private fun poseForward(pose: Pose): ArGeometry.Vec3 {
        val z = pose.zAxis
        return ArGeometry.Vec3(-z[0].toDouble(), -z[1].toDouble(), -z[2].toDouble())
    }

    private fun finiteOrNull(value: Double?): Any {
        if (value == null || !value.isFinite()) return JSONObject.NULL
        return value
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
        } catch (_: Exception) {
        }
    }

    override fun onDestroy() {
        try {
            baseAnchorNode?.anchor?.detach()
            baseAnchorNode?.setParent(null)
        } catch (_: Exception) {
        }
        super.onDestroy()
    }
}
