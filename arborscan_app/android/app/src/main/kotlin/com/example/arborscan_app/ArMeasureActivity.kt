// ArborScan AR measurement screen with manual / automatic point placement modes.
package com.example.arborscan_app

import android.app.Activity
import android.content.Intent
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.widget.ImageButton
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.ar.core.Anchor
import com.google.ar.core.Frame
import com.google.ar.core.HitResult
import com.google.ar.core.Plane
import com.google.ar.core.Point
import com.google.ar.core.TrackingState
import com.google.ar.sceneform.AnchorNode
import com.google.ar.sceneform.FrameTime
import com.google.ar.sceneform.math.Vector3
import com.google.ar.sceneform.rendering.Color as SceneColor
import com.google.ar.sceneform.rendering.MaterialFactory
import com.google.ar.sceneform.rendering.ShapeFactory
import com.google.ar.sceneform.ux.ArFragment
import org.json.JSONObject
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class ArMeasureActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_RESULT_JSON = "result_json"
        private const val MAX_POINTS = 6

        private const val AUTO_PLACE_STABLE_MS = 800L
        private const val AUTO_PLACE_COOLDOWN_MS = 900L
        private const val STABLE_HIT_MAX_WORLD_DELTA_M = 0.03f
    }

    private lateinit var arFragment: ArFragment
    private lateinit var hintText: TextView
    private lateinit var statusText: TextView
    private lateinit var progressText: TextView
    private lateinit var zoomText: TextView
    private lateinit var modeBtn: TextView
    private lateinit var placeBtn: ImageButton
    private lateinit var undoBtn: ImageButton
    private lateinit var doneBtn: ImageButton
    private lateinit var zoomInBtn: ImageButton
    private lateinit var zoomOutBtn: ImageButton
    private lateinit var reticleView: View

    private val anchorNodes = mutableListOf<AnchorNode>()
    private var currentHit: HitResult? = null
    private var currentUsesFeaturePoint = false
    private var centerReady = false
    private var lastPlacementUsedFeaturePoint = false

    // false by default: field measurements are safer when the user confirms every point.
    private var autoPlacementEnabled = false

    private var stableHitSinceMs: Long = 0L
    private var lastAutoPlaceMs: Long = 0L
    private var lastHitSample: Vector3? = null

    // Aim-assist zoom. Replace with true camera zoom here if your AR stack exposes it.
    private var zoomAssist = 1.0f
    private val zoomMin = 1.0f
    private val zoomMax = 4.0f

    private lateinit var scaleDetector: ScaleGestureDetector

    // ML
    private lateinit var tflite: TFLiteHelper
    private lateinit var treeDetector: TreeDetector
    private var lastMlTime = 0L
    private val ML_INTERVAL = 500L
    private var isProcessing = false
    private val mlExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ar_measure)

        arFragment = supportFragmentManager.findFragmentById(R.id.arFragment) as ArFragment
        hintText = findViewById(R.id.tvHint)
        statusText = findViewById(R.id.tvStatus)
        progressText = findViewById(R.id.tvProgress)
        zoomText = findViewById(R.id.tvZoom)
        modeBtn = findViewById(R.id.btnPlacementMode)
        placeBtn = findViewById(R.id.btnPlace)
        undoBtn = findViewById(R.id.btnUndo)
        doneBtn = findViewById(R.id.btnDone)
        zoomInBtn = findViewById(R.id.btnZoomIn)
        zoomOutBtn = findViewById(R.id.btnZoomOut)
        reticleView = findViewById(R.id.reticle)

        scaleDetector = ScaleGestureDetector(
            this,
            object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
                override fun onScale(detector: ScaleGestureDetector): Boolean {
                    setZoomAssist(zoomAssist * detector.scaleFactor)
                    return true
                }
            }
        )

        arFragment.arSceneView.scene.addOnUpdateListener(this::onSceneUpdate)
        arFragment.setOnTapArPlaneListener { _, _, _ -> }

        modeBtn.setOnClickListener { togglePlacementMode() }
        placeBtn.setOnClickListener { placeCenterPoint(isAuto = false) }
        undoBtn.setOnClickListener { undoLastPoint() }
        doneBtn.setOnClickListener {
            android.util.Log.d("AR_DEBUG", "DONE CLICKED, points=$anchorNodes.size")
            finishMeasure()
        }
        zoomInBtn.setOnClickListener { setZoomAssist(zoomAssist + 0.2f) }
        zoomOutBtn.setOnClickListener { setZoomAssist(zoomAssist - 0.2f) }

        updateZoomLabel()
        updatePlacementModeLabel()

        // ML init
        tflite = TFLiteHelper(this)
        treeDetector = TreeDetector(tflite)

        updateUi()
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        scaleDetector.onTouchEvent(ev)
        return super.dispatchTouchEvent(ev)
    }

    private fun onSceneUpdate(frameTime: FrameTime) {
        val frame = arFragment.arSceneView.arFrame ?: return

        val now = System.currentTimeMillis()
        if (!isProcessing && now - lastMlTime > ML_INTERVAL) {
            processFrame(frame)
            lastMlTime = now
        }

        if (frame.camera.trackingState != TrackingState.TRACKING) {
            currentHit = null
            centerReady = false
            currentUsesFeaturePoint = false
            stableHitSinceMs = 0L
            lastHitSample = null
            updateUi()
            return
        }

        val hitInfo = findAdaptiveCenterHit(frame)
        currentHit = hitInfo?.first
        currentUsesFeaturePoint = hitInfo?.second == true
        centerReady = currentHit != null

        updateStableHitState()
        if (autoPlacementEnabled) {
            maybeAutoPlace()
        }

        updateUi()
    }

    private fun currentStageIndex(): Int = anchorNodes.size.coerceIn(0, MAX_POINTS)

    private fun stageScreenBias(viewWidth: Float, viewHeight: Float): Pair<Float, Float> {
        val stage = currentStageIndex()
        return when (stage) {
            0 -> 0f to (viewHeight * 0.18f)     // основание дерева: ниже центра
            1 -> 0f to (-viewHeight * 0.22f)    // верхушка: выше центра
            2 -> (-viewWidth * 0.18f) to (-viewHeight * 0.05f) // левая крона
            3 -> (viewWidth * 0.18f) to (-viewHeight * 0.05f)  // правая крона
            4 -> (-viewWidth * 0.06f) to (viewHeight * 0.10f)  // левый край ствола
            5 -> (viewWidth * 0.06f) to (viewHeight * 0.10f)   // правый край ствола
            else -> 0f to 0f
        }
    }

    private fun findAdaptiveCenterHit(frame: Frame): Pair<HitResult, Boolean>? {
        val view = arFragment.arSceneView
        val centerX = view.width / 2f
        val centerY = view.height / 2f
        val (biasX, biasY) = stageScreenBias(view.width.toFloat(), view.height.toFloat())

        val samples = listOf(
            Pair(centerX + biasX, centerY + biasY),
            Pair(centerX + biasX, centerY + biasY - 24f),
            Pair(centerX + biasX, centerY + biasY + 24f),
            Pair(centerX + biasX - 24f, centerY + biasY),
            Pair(centerX + biasX + 24f, centerY + biasY),
            Pair(centerX, centerY)
        )

        var featureFallback: HitResult? = null

        for ((x, y) in samples) {
            val hits = frame.hitTest(x, y)
            for (hit in hits) {
                val trackable = hit.trackable
                when (trackable) {
                    is Plane -> {
                        if (trackable.isPoseInPolygon(hit.hitPose)) {
                            return hit to false
                        }
                    }
                    is Point -> {
                        if (trackable.orientationMode == Point.OrientationMode.ESTIMATED_SURFACE_NORMAL) {
                            if (featureFallback == null) featureFallback = hit
                        }
                    }
                }
            }
        }

        return featureFallback?.let { it to true }
    }

    private fun updateStableHitState() {
        val hit = currentHit ?: run {
            stableHitSinceMs = 0L
            lastHitSample = null
            return
        }

        val p = Vector3(hit.hitPose.tx(), hit.hitPose.ty(), hit.hitPose.tz())
        val now = System.currentTimeMillis()
        val prev = lastHitSample

        if (prev == null) {
            lastHitSample = p
            stableHitSinceMs = now
            return
        }

        val delta = worldDistance(prev, p)
        if (delta <= STABLE_HIT_MAX_WORLD_DELTA_M) {
            if (stableHitSinceMs == 0L) stableHitSinceMs = now
        } else {
            stableHitSinceMs = now
        }
        lastHitSample = p
    }

    private fun maybeAutoPlace() {
        if (!centerReady) return
        if (anchorNodes.size >= MAX_POINTS) return

        val now = System.currentTimeMillis()
        if (stableHitSinceMs == 0L) return
        if (now - stableHitSinceMs < AUTO_PLACE_STABLE_MS) return
        if (now - lastAutoPlaceMs < AUTO_PLACE_COOLDOWN_MS) return

        placeCenterPoint(isAuto = true)
        lastAutoPlaceMs = System.currentTimeMillis()
        stableHitSinceMs = 0L
    }

    private fun placeCenterPoint(isAuto: Boolean) {
        if (anchorNodes.size >= MAX_POINTS) {
            statusText.text = "Уже поставлено 6 точек. Нажми Готово или Undo."
            updateUi()
            return
        }

        val hit = currentHit
        if (hit == null) {
            statusText.text = "Не удалось найти поверхность. Подойди ближе или наведи на более контрастную часть объекта."
            updateUi()
            return
        }

        lastPlacementUsedFeaturePoint = currentUsesFeaturePoint

        val anchor = hit.createAnchor()
        addAnchorNode(anchor)

        when (anchorNodes.size) {
            2 -> statusText.text = "Высота сохранена: %.2f м".format(segmentDistance(0, 1))
            4 -> statusText.text = "Ширина кроны сохранена: %.2f м".format(segmentDistance(2, 3))
            6 -> {
                val h = segmentDistance(0, 1)
                val c = segmentDistance(2, 3)
                val t = segmentDistance(4, 5)
                statusText.text = "Высота: %.2f м\nКрона: %.2f м\nСтвол: %.2f м".format(h, c, t)
            }
            else -> {
                val prefix = if (isAuto) "Точка поставлена автоматически. " else "Точка поставлена вручную. "
                statusText.text = prefix + currentStageDescription()
            }
        }

        stableHitSinceMs = 0L
        lastHitSample = null
        updateUi()
    }

    private fun addAnchorNode(anchor: Anchor) {
        val anchorNode = AnchorNode(anchor).apply {
            setParent(arFragment.arSceneView.scene)
        }

        val pointColor = when {
            anchorNodes.size < 2 -> SceneColor(0.27f, 0.88f, 0.63f)
            anchorNodes.size < 4 -> SceneColor(0.96f, 0.69f, 0.24f)
            else -> SceneColor(1.0f, 0.42f, 0.42f)
        }

        MaterialFactory.makeOpaqueWithColor(this, pointColor)
            .thenAccept { material ->
                val sphere = ShapeFactory.makeSphere(
                    0.015f,
                    Vector3.zero(),
                    material
                )
                anchorNode.renderable = sphere
            }

        anchorNodes.add(anchorNode)
    }

    private fun undoLastPoint() {
        val last = if (anchorNodes.isNotEmpty()) anchorNodes.removeAt(anchorNodes.lastIndex) else null
        if (last == null) return

        last.anchor?.detach()
        last.setParent(null)
        statusText.text = "Последняя точка удалена"
        stableHitSinceMs = 0L
        lastHitSample = null
        updateUi()
    }

    private fun finishMeasure() {
        if (anchorNodes.size < MAX_POINTS) {
            statusText.text = "Поставь все 6 точек: 2 для высоты, 2 для кроны, 2 для ствола."
            updateUi()
            return
        }

        val height = segmentDistance(0, 1)
        val crown = segmentDistance(2, 3)
        val trunk = segmentDistance(4, 5)

        val json = JSONObject()
            .put("height_m", height)
            .put("height_cm", height * 100.0)
            .put("crown_width_m", crown)
            .put("trunk_diameter_m", trunk)
            .put("distance_m", height)
            .put("distance_cm", height * 100.0)
            .put("points_count", anchorNodes.size)
            .put("zoom_assist", zoomAssist.toDouble())
            .put("used_feature_point", lastPlacementUsedFeaturePoint)
            .put("center_placement", true)
            .put("placement_mode", if (autoPlacementEnabled) "auto" else "manual")
            .toString()

        android.util.Log.d("AR_RESULT", json)

        val data = Intent().putExtra(EXTRA_RESULT_JSON, json)
        setResult(Activity.RESULT_OK, data)
        finish()
    }

    private fun segmentDistance(aIndex: Int, bIndex: Int): Double {
        if (anchorNodes.size <= bIndex) return 0.0
        return distanceBetween(anchorNodes[aIndex].anchor, anchorNodes[bIndex].anchor)
    }

    private fun currentStageDescription(): String {
        return when (anchorNodes.size) {
            0 -> "Этап 1/3: наведи центр на нижнюю точку дерева"
            1 -> "Этап 1/3: наведи центр на верхнюю точку дерева"
            2 -> "Этап 2/3: наведи центр на левую границу кроны"
            3 -> "Этап 2/3: наведи центр на правую границу кроны"
            4 -> "Этап 3/3: наведи центр на левую границу ствола"
            5 -> "Этап 3/3: наведи центр на правую границу ствола"
            else -> "Все точки поставлены"
        }
    }

    private fun updateUi() {
        val tracking = arFragment.arSceneView.arFrame?.camera?.trackingState == TrackingState.TRACKING
        progressText.text = "Точек: ${anchorNodes.size}/6"
        updatePlacementModeLabel()

        when {
            !tracking -> {
                hintText.text = "Двигай телефон медленно, пока AR не стабилизируется"
                statusText.text = "Трекинг ещё не готов"
                setReticleColor("#FF6B6B")
            }
            anchorNodes.size >= MAX_POINTS -> {
                hintText.text = "Все 6 точек поставлены"
                val h = segmentDistance(0, 1)
                val c = segmentDistance(2, 3)
                val t = segmentDistance(4, 5)
                statusText.text = "Высота: %.2f м\nКрона: %.2f м\nСтвол: %.2f м".format(h, c, t)
                setReticleColor("#46E0A1")
            }
            centerReady && currentUsesFeaturePoint -> {
                hintText.text = currentStageDescription()
                statusText.text = if (autoPlacementEnabled) {
                    "Наведение есть через feature point. Удерживай центр ровно или нажми кнопку."
                } else {
                    "Наведение есть через feature point. Нажми +, если точка выбрана правильно."
                }
                setReticleColor("#F4B03E")
            }
            centerReady -> {
                hintText.text = currentStageDescription()
                statusText.text = if (autoPlacementEnabled) {
                    val now = System.currentTimeMillis()
                    val stableMs = if (stableHitSinceMs == 0L) 0L else now - stableHitSinceMs
                    val remain = max(0L, AUTO_PLACE_STABLE_MS - stableMs)
                    if (remain > 0) {
                        "Авто: удерживай центр ещё ${"%.1f".format(remain / 1000f)} с или нажми кнопку."
                    } else {
                        "Авто: центр стабилен, точка будет поставлена автоматически."
                    }
                } else {
                    "Ручной режим: наведи центр и нажми +, чтобы поставить точку."
                }
                setReticleColor("#46E0A1")
            }
            else -> {
                hintText.text = currentStageDescription()
                statusText.text = "AR не видит достаточно точек на объекте"
                setReticleColor("#FF6B6B")
            }
        }

        val ready = anchorNodes.size >= MAX_POINTS
        undoBtn.isEnabled = anchorNodes.isNotEmpty()
        placeBtn.isEnabled = centerReady && anchorNodes.size < MAX_POINTS
        doneBtn.isEnabled = ready
        doneBtn.alpha = if (ready) 1.0f else 0.4f
    }

    private fun togglePlacementMode() {
        autoPlacementEnabled = !autoPlacementEnabled
        stableHitSinceMs = 0L
        lastAutoPlaceMs = 0L
        lastHitSample = null
        statusText.text = if (autoPlacementEnabled) {
            "Включён автоматический режим: точка ставится после стабильного наведения."
        } else {
            "Включён ручной режим: каждая точка ставится кнопкой +."
        }
        updatePlacementModeLabel()
        updateUi()
    }

    private fun updatePlacementModeLabel() {
        if (!::modeBtn.isInitialized) return
        val bgColor = if (autoPlacementEnabled) "#3346E0A1" else "#33132238"
        val strokeColor = if (autoPlacementEnabled) "#46E0A1" else "#6F7F91"
        val textColor = if (autoPlacementEnabled) "#46E0A1" else "#F6FAFF"

        modeBtn.text = if (autoPlacementEnabled) "Режим: авто" else "Режим: ручной"
        modeBtn.setTextColor(Color.parseColor(textColor))
        modeBtn.background = GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = 28f
            setColor(Color.parseColor(bgColor))
            setStroke(3, Color.parseColor(strokeColor))
        }
    }

    private fun setReticleColor(hex: String) {
        val color = Color.parseColor(hex)
        val d = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(Color.TRANSPARENT)
            setStroke(5, color)
        }
        reticleView.background = d
    }

    private fun setZoomAssist(value: Float) {
        zoomAssist = min(zoomMax, max(zoomMin, value))
        updateZoomLabel()

        arFragment.requireView().apply {
            pivotX = width / 2f
            pivotY = height / 2f
            scaleX = zoomAssist
            scaleY = zoomAssist
        }
    }

    private fun updateZoomLabel() {
        zoomText.text = "ZOOM x%.1f".format(zoomAssist)
    }

    private fun distanceBetween(a: Anchor?, b: Anchor?): Double {
        if (a == null || b == null) return 0.0
        val ap = a.pose
        val bp = b.pose
        val dx = ap.tx() - bp.tx()
        val dy = ap.ty() - bp.ty()
        val dz = ap.tz() - bp.tz()
        return sqrt(dx * dx + dy * dy + dz * dz).toDouble()
    }

    private fun worldDistance(a: Vector3, b: Vector3): Float {
        val dx = a.x - b.x
        val dy = a.y - b.y
        val dz = a.z - b.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    // ================= ML =================

    private fun processFrame(frame: Frame) {
        if (isProcessing) return
        isProcessing = true

        try {
            val image = frame.acquireCameraImage()

            mlExecutor.execute {
                try {
                    val bitmap = imageToBitmap(image)
                    val resized = android.graphics.Bitmap.createScaledBitmap(bitmap, 320, 320, true)

                    val result = treeDetector.detect(resized)

                    if (result.hasTree) {
                        android.util.Log.d("ML", "🌳 Дерево найдено")
                    } else {
                        android.util.Log.d("ML", "❌ Нет дерева")
                    }

                } catch (e: Exception) {
                    android.util.Log.e("ML", "Ошибка ML: ${e.message}")
                } finally {
                    image.close()
                    isProcessing = false
                }
            }

        } catch (e: Exception) {
            isProcessing = false
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
        anchorNodes.forEach {
            it.anchor?.detach()
            it.setParent(null)
        }
        anchorNodes.clear()
        mlExecutor.shutdownNow()
        super.onDestroy()
    }
}
