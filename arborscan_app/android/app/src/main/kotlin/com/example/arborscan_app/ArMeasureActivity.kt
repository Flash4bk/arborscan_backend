package com.example.arborscan_app

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.os.SystemClock
import java.io.FileOutputStream
import java.io.File
import android.view.PixelCopy
import android.os.HandlerThread
import android.os.Handler
import android.graphics.Bitmap
import android.view.MotionEvent
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.ar.core.Anchor
import com.google.ar.core.Frame
import com.google.ar.core.HitResult
import com.google.ar.core.Plane
import com.google.ar.core.Point
import com.google.ar.core.Trackable
import com.google.ar.core.TrackingFailureReason
import com.google.ar.core.TrackingState
import com.google.ar.sceneform.AnchorNode
import com.google.ar.sceneform.Node
import com.google.ar.sceneform.math.Vector3
import com.google.ar.sceneform.rendering.Color
import com.google.ar.sceneform.rendering.MaterialFactory
import com.google.ar.sceneform.rendering.ShapeFactory
import com.google.ar.sceneform.ux.ArFragment
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.sqrt

class ArMeasureActivity : AppCompatActivity() {

    private lateinit var arFragment: ArFragment
    private lateinit var statusText: TextView
    private lateinit var doneBtn: Button
    private lateinit var resetBtn: Button
    private lateinit var undoBtn: Button

    // World points in meters (ARCore world space)
    private val points = mutableListOf<Vector3>()

    // Keep created anchors so we can undo/reset precisely
    private val anchorNodes = mutableListOf<AnchorNode>()

    // Diagnostics for "real assistant" tips
    private var pointHitsCount: Int = 0
    private var planeHitsCount: Int = 0

    // 2 / 4 / 6 points (default: full 6)
    private val requiredPoints: Int by lazy {
        val v = intent.getIntExtra("required_points", 6)
        when (v) {
            2, 4, 6 -> v
            else -> 6
        }
    }

    // Throttle assistant UI updates to avoid flicker
    private var lastAssistantUiAtMs: Long = 0L
    private var lastTrackingState: TrackingState? = null
    private var lastFailureReason: TrackingFailureReason? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ar_measure)

        arFragment = supportFragmentManager.findFragmentById(R.id.ar_fragment) as ArFragment
        statusText = findViewById(R.id.status_text)
        doneBtn = findViewById(R.id.done_btn)
        resetBtn = findViewById(R.id.reset_btn)
        undoBtn = findViewById(R.id.undo_btn)

        // Visualize planes (ground)
        arFragment.arSceneView.planeRenderer.isVisible = true

        // Real assistant: watch tracking state and adapt hints
        arFragment.arSceneView.scene.addOnUpdateListener {
            val frame = arFragment.arSceneView.arFrame ?: return@addOnUpdateListener
            maybeUpdateAssistantUi(frame)
        }

        updateStatus(force = true)

        // Universal tap: use frame.hitTest(motionEvent) so we can hit both Plane and Point.
        arFragment.arSceneView.scene.addOnPeekTouchListener { _, motionEvent ->
            if (motionEvent.action != MotionEvent.ACTION_UP) return@addOnPeekTouchListener

            // If we've already collected required points, ignore taps (avoid accidental extra points)
            if (points.size >= requiredPoints) return@addOnPeekTouchListener

            val frame = arFragment.arSceneView.arFrame ?: return@addOnPeekTouchListener
            if (frame.camera.trackingState != TrackingState.TRACKING) {
                // Give immediate assistant feedback
                updateStatus(force = true)
                return@addOnPeekTouchListener
            }

            val hits = frame.hitTest(motionEvent)
            val hit = pickBestHit(hits, points.size) ?: return@addOnPeekTouchListener
            placeAnchor(hit)
        }

        undoBtn.setOnClickListener { undoLastPoint() }
        resetBtn.setOnClickListener { resetAll() }

        doneBtn.setOnClickListener {
    // Capture a snapshot of current AR view so Flutter can run full /analyze-tree (species/risk) immediately.
    doneBtn.isEnabled = false
    statusText.text = "Сохраняю замер и кадр…"

    captureArSceneToCacheJpeg { capturePath ->
        runOnUiThread {
            val result = buildResultJson()
            if (capturePath != null) {
                result.put("capture_path", capturePath)
            }
            val intent = Intent()
            intent.putExtra("result_json", result.toString())
            setResult(Activity.RESULT_OK, intent)
            finish()
        }
    }
}
    }

    /**
     * Pick best hit depending on step:
     * - For first point (index 0): prefer Plane (ground).
     * - For the rest: prefer Point (feature/depth points) so taps on tree don't fall to ground/background.
     */
    private fun pickBestHit(hits: List<HitResult>, pointIndex: Int): HitResult? {
        if (hits.isEmpty()) return null
        val preferPlane = (pointIndex == 0)

        fun isTracking(t: Trackable): Boolean =
            t.trackingState == TrackingState.TRACKING

        // Pass 1: preferred trackables first
        for (h in hits) {
            val t = h.trackable
            if (!isTracking(t)) continue

            if (!preferPlane) {
                // Prefer points (tree/crown)
                if (t is Point && t.orientationMode == Point.OrientationMode.ESTIMATED_SURFACE_NORMAL) return h
                if (t is Point) return h
            } else {
                // First point: ground plane
                if (t is Plane && t.isPoseInPolygon(h.hitPose)) return h
            }
        }

        // Pass 2: fallback to anything reasonable
        for (h in hits) {
            val t = h.trackable
            if (!isTracking(t)) continue
            if (t is Plane && t.isPoseInPolygon(h.hitPose)) return h
            if (t is Point) return h
        }

        return hits.firstOrNull()
    }

    private fun placeAnchor(hit: HitResult) {
        val anchor: Anchor = hit.createAnchor()
        val anchorNode = AnchorNode(anchor)
        anchorNode.setParent(arFragment.arSceneView.scene)
        anchorNodes.add(anchorNode)

        // Diagnostics for assistant tips (helps explain "why it might be off")
        when (hit.trackable) {
            is Point -> pointHitsCount++
            is Plane -> planeHitsCount++
        }

        val pose = anchor.pose
        val p = Vector3(pose.tx(), pose.ty(), pose.tz())
        points.add(p)

        addMarker(anchorNode)
        updateStatus(force = true)
    }

    private fun addMarker(parent: AnchorNode) {
        MaterialFactory.makeOpaqueWithColor(this, Color(0.1f, 0.9f, 0.2f))
            .thenAccept { material ->
                val sphere = ShapeFactory.makeSphere(0.03f, Vector3.zero(), material)
                val node = Node()
                node.renderable = sphere
                node.setParent(parent)
                node.localPosition = Vector3.zero()
            }
    }

    private fun undoLastPoint() {
        if (points.isEmpty()) return
        points.removeAt(points.size - 1)

        val scene = arFragment.arSceneView.scene
        val lastNode = anchorNodes.removeLastOrNull()
        if (lastNode != null) {
            lastNode.anchor?.detach()
            scene.removeChild(lastNode)
        }

        // Keep diagnostics roughly consistent (best-effort)
        // If user is undoing, we can't know exact hit type; don't decrement below 0.
        if (pointHitsCount > 0) pointHitsCount--
        if (planeHitsCount > 0 && points.size == 0) planeHitsCount-- // usually first point was plane

        updateStatus(force = true)
    }

    private fun resetAll() {
        points.clear()
        val scene = arFragment.arSceneView.scene
        for (n in anchorNodes) {
            n.anchor?.detach()
            scene.removeChild(n)
        }
        anchorNodes.clear()
        pointHitsCount = 0
        planeHitsCount = 0
        updateStatus(force = true)
    }

    // Height is better measured along vertical axis (Y)
    private fun heightMeters(a: Vector3, b: Vector3): Double =
        abs((a.y - b.y).toDouble())

    // Trunk diameter / crown width: prefer horizontal component (XZ)
    private fun horizontalMeters(a: Vector3, b: Vector3): Double {
        val dx = (a.x - b.x).toDouble()
        val dz = (a.z - b.z).toDouble()
        return sqrt(dx * dx + dz * dz)
    }

    private fun fmt(m: Double): String = String.format("%.2f м", m)

    private fun modeLabel(): String = when (requiredPoints) {
        2 -> "Высота (2 точки)"
        4 -> "Высота + диаметр (4 точки)"
        else -> "Полный замер (6 точек)"
    }

    private fun primaryCtaLabel(n: Int): String = when {
        n >= requiredPoints -> "Готово: нажми «Сохранить»"
        n == 0 -> "Поставь 1-ю точку у основания (на земле)"
        n == 1 -> "Поставь 2-ю точку на верхушке"
        requiredPoints >= 4 && n == 2 -> "Поставь 3-ю точку на краю ствола (~1.3м)"
        requiredPoints >= 4 && n == 3 -> "Поставь 4-ю точку на другом краю ствола"
        requiredPoints >= 6 && n == 4 -> "Поставь 5-ю точку на краю кроны"
        requiredPoints >= 6 && n == 5 -> "Поставь 6-ю точку на другом краю кроны"
        else -> "Поставь следующую точку"
    }

    private fun assistantTrackingTip(frame: Frame): String? {
        val cam = frame.camera
        if (cam.trackingState == TrackingState.TRACKING) return null

        val reason = cam.trackingFailureReason
        return when (reason) {
            TrackingFailureReason.INSUFFICIENT_LIGHT ->
                "Трекинг слабый: мало света. Включи фонарик/подойди к свету."
            TrackingFailureReason.EXCESSIVE_MOTION ->
                "Трекинг слабый: слишком резкие движения. Двигай телефоном медленнее."
            TrackingFailureReason.INSUFFICIENT_FEATURES ->
                "Трекинг слабый: мало деталей. Наведи на контраст (кора/ветки), отойди на 2–3 шага."
            TrackingFailureReason.CAMERA_UNAVAILABLE ->
                "Камера недоступна: закрой другие приложения с камерой."
            else ->
                "Трекинг не готов. Поводи камерой и подожди 2–3 сек."
        }
    }

    private fun assistantQualityTip(): String? {
        // After first point, we want points on the tree (Point hits)
        if (points.size >= 2) {
            if (planeHitsCount >= 2 && pointHitsCount == 0) {
                return "Похоже, точки цепляются за землю/фон. Для точек 2+ тапай по дереву (ствол/крона)."
            }
        }
        return null
    }

    private fun plausibilityWarning(height: Double?, trunk: Double?, crown: Double?): String? {
        if (height != null && (height < 0.5 || height > 80.0)) return "⚠️ Высота выглядит неверно — проверь точки 1–2"
        if (trunk != null && (trunk < 0.03 || trunk > 3.0)) return "⚠️ Диаметр выглядит неверно — проверь точки 3–4"
        if (crown != null && (crown < 0.2 || crown > 60.0)) return "⚠️ Крона выглядит неверно — проверь точки 5–6"
        return null
    }

    private fun maybeUpdateAssistantUi(frame: Frame) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastAssistantUiAtMs < 250) return // 4 updates/sec max

        val cam = frame.camera
        val state = cam.trackingState
        val reason = cam.trackingFailureReason

        // Update only on meaningful changes to reduce flicker
        if (state != lastTrackingState || reason != lastFailureReason) {
            lastTrackingState = state
            lastFailureReason = reason
            updateStatus(force = true)
            lastAssistantUiAtMs = now
        }
    }

    private fun updateStatus(force: Boolean = false) {
        // 'force' kept for future expansions; currently always updates.
        val n = points.size

        val height = if (n >= 2) heightMeters(points[0], points[1]) else null
        val trunk = if (n >= 4) horizontalMeters(points[2], points[3]) else null
        val crown = if (n >= 6) horizontalMeters(points[4], points[5]) else null

        val frame = arFragment.arSceneView.arFrame
        val trackingTip = if (frame != null) assistantTrackingTip(frame) else null
        val qualityTip = assistantQualityTip()
        val warn = plausibilityWarning(height, trunk, crown)

        // "Real assistant": one primary CTA line + optional one-liners
        val primary = if (trackingTip != null) trackingTip else primaryCtaLabel(n)

        val lines = StringBuilder()
        lines.append("Помощник: ").append(primary).append('\n')
        lines.append("Режим: ").append(modeLabel()).append('\n')

        if (qualityTip != null) lines.append(qualityTip).append('\n')
        if (warn != null) lines.append(warn).append('\n')

        lines.append('\n')
        lines.append("Высота: ").append(height?.let { fmt(it) } ?: "—").append('\n')
        if (requiredPoints >= 4) {
            lines.append("Диаметр ствола: ").append(trunk?.let { fmt(it) } ?: "—").append('\n')
        } else {
            lines.append("Диаметр ствола: —").append('\n')
        }
        if (requiredPoints >= 6) {
            lines.append("Ширина кроны: ").append(crown?.let { fmt(it) } ?: "—").append('\n')
        } else {
            lines.append("Ширина кроны: —").append('\n')
        }

        // Tiny diagnostic (helps you during testing; can be removed later)
        if (n > 0) {
            lines.append('\n')
            lines.append("Диагн.: point=").append(pointHitsCount).append(", plane=").append(planeHitsCount)
        }

        statusText.text = lines.toString()

        doneBtn.isEnabled = (n >= requiredPoints)
        undoBtn.isEnabled = (n > 0)
        resetBtn.isEnabled = (n > 0)
    }

    private fun captureArSceneToCacheJpeg(onDone: (String?) -> Unit) {
    val view = arFragment.arSceneView
    if (view.width <= 0 || view.height <= 0) {
        onDone(null)
        return
    }

    val bitmap = Bitmap.createBitmap(view.width, view.height, Bitmap.Config.ARGB_8888)
    val thread = HandlerThread("PixelCopy")
    thread.start()
    val handler = Handler(thread.looper)

    PixelCopy.request(view, bitmap, { copyResult ->
        try {
            if (copyResult == PixelCopy.SUCCESS) {
                val outFile = File(cacheDir, "ar_capture_${System.currentTimeMillis()}.jpg")
                FileOutputStream(outFile).use { fos ->
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 85, fos)
                }
                onDone(outFile.absolutePath)
            } else {
                onDone(null)
            }
        } catch (_: Throwable) {
            onDone(null)
        } finally {
            thread.quitSafely()
        }
    }, handler)
}

private fun buildResultJson(): JSONObject {
        val obj = JSONObject()
        obj.put("points_count", points.size)
        obj.put("required_points", requiredPoints)

        if (points.size >= 2) obj.put("height_m", heightMeters(points[0], points[1]))
        if (points.size >= 4) obj.put("trunk_diameter_m", horizontalMeters(points[2], points[3]))
        if (points.size >= 6) obj.put("crown_width_m", horizontalMeters(points[4], points[5]))

        // Assistant diagnostics (optional; useful for QA and for backend confidence later)
        obj.put("point_hits_count", pointHitsCount)
        obj.put("plane_hits_count", planeHitsCount)

        return obj
    }
}