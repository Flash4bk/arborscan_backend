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
    }

    private lateinit var arFragment: ArFragment
    private lateinit var hintText: TextView
    private lateinit var statusText: TextView
    private lateinit var progressText: TextView
    private lateinit var zoomText: TextView
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

    // Aim-assist zoom. Replace with true camera zoom here if your AR stack exposes it.
    private var zoomAssist = 1.0f
    private val zoomMin = 1.0f
    private val zoomMax = 4.0f

    private lateinit var scaleDetector: ScaleGestureDetector

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ar_measure)

        arFragment = supportFragmentManager.findFragmentById(R.id.arFragment) as ArFragment
        hintText = findViewById(R.id.tvHint)
        statusText = findViewById(R.id.tvStatus)
        progressText = findViewById(R.id.tvProgress)
        zoomText = findViewById(R.id.tvZoom)
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

        placeBtn.setOnClickListener { placeCenterPoint() }
        undoBtn.setOnClickListener { undoLastPoint() }
        doneBtn.setOnClickListener {
            android.util.Log.d("AR_DEBUG", "DONE CLICKED, points=" + anchorNodes.size)
            finishMeasure()
        }
        zoomInBtn.setOnClickListener { setZoomAssist(zoomAssist + 0.2f) }
        zoomOutBtn.setOnClickListener { setZoomAssist(zoomAssist - 0.2f) }

        updateZoomLabel()
        updateUi()
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        scaleDetector.onTouchEvent(ev)
        return super.dispatchTouchEvent(ev)
    }

    private fun onSceneUpdate(frameTime: FrameTime) {
        val frame = arFragment.arSceneView.arFrame ?: return

        if (frame.camera.trackingState != TrackingState.TRACKING) {
            currentHit = null
            centerReady = false
            currentUsesFeaturePoint = false
            updateUi()
            return
        }

        val hitInfo = findCenterHit(frame)
        currentHit = hitInfo?.first
        currentUsesFeaturePoint = hitInfo?.second == true
        centerReady = currentHit != null
        updateUi()
    }

    private fun findCenterHit(frame: Frame): Pair<HitResult, Boolean>? {
        val view = arFragment.arSceneView
        val x = view.width / 2f
        val y = view.height / 2f

        val hits = frame.hitTest(x, y)
        var featureFallback: HitResult? = null

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
                        if (featureFallback == null) {
                            featureFallback = hit
                        }
                    }
                }
            }
        }
        return featureFallback?.let { it to true }
    }

    private fun placeCenterPoint() {
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
            else -> statusText.text = currentStageDescription()
        }

        updateUi()
    }

    private fun addAnchorNode(anchor: Anchor) {
        val anchorNode = AnchorNode(anchor).apply {
            setParent(arFragment.arSceneView.scene)
        }

        val pointColor = when {
            anchorNodes.size < 2 -> SceneColor(0.27f, 0.88f, 0.63f) // height
            anchorNodes.size < 4 -> SceneColor(0.96f, 0.69f, 0.24f) // crown
            else -> SceneColor(1.0f, 0.42f, 0.42f) // trunk
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
            .put("distance_m", height) // backward compatibility
            .put("distance_cm", height * 100.0)
            .put("points_count", anchorNodes.size)
            .put("zoom_assist", zoomAssist.toDouble())
            .put("used_feature_point", lastPlacementUsedFeaturePoint)
            .put("center_placement", true)
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

        when {
            !tracking -> {
                hintText.text = "Двигай телефон медленно, пока AR не стабилизируется"
                statusText.text = "Трекинг ещё не готов"
                setReticleColor("#FF6B6B")
            }
            anchorNodes.size >= MAX_POINTS -> {
                hintText.text = "Все 6 точек поставлены"
                statusText.text = if (anchorNodes.size >= MAX_POINTS) {
                    val h = segmentDistance(0, 1)
                    val c = segmentDistance(2, 3)
                    val t = segmentDistance(4, 5)
                    "Высота: %.2f м\nКрона: %.2f м\nСтвол: %.2f м".format(h, c, t)
                } else {
                    "Нажми Готово, чтобы вернуть размеры в приложение"
                }
                setReticleColor("#46E0A1")
            }
            centerReady && currentUsesFeaturePoint -> {
                hintText.text = currentStageDescription()
                statusText.text = "Fallback: feature point. Можно ставить, но plane был бы точнее."
                setReticleColor("#F4B03E")
            }
            centerReady -> {
                hintText.text = currentStageDescription()
                statusText.text = "Центр готов. Нажми кнопку, чтобы поставить точку."
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

        // Aim-assist zoom
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

    override fun onDestroy() {
        anchorNodes.forEach {
            it.anchor?.detach()
            it.setParent(null)
        }
        anchorNodes.clear()
        super.onDestroy()
    }
}
