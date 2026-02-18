package com.example.arborscan_app

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.MotionEvent
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.google.ar.core.Config
import com.google.ar.core.DepthPoint
import com.google.ar.core.Frame
import com.google.ar.core.HitResult
import com.google.ar.core.InstantPlacementPoint
import com.google.ar.core.Plane
import com.google.ar.core.Point
import com.google.ar.core.TrackingState
import com.google.ar.sceneform.AnchorNode
import com.google.ar.sceneform.Node
import com.google.ar.sceneform.math.Vector3
import com.google.ar.sceneform.rendering.Color
import com.google.ar.sceneform.rendering.MaterialFactory
import com.google.ar.sceneform.rendering.ShapeFactory
import com.google.ar.sceneform.ux.ArFragment
import org.json.JSONObject
import kotlin.math.sqrt

class ArMeasureActivity : AppCompatActivity() {

    private lateinit var arFragment: ArFragment
    private lateinit var statusText: TextView
    private lateinit var doneBtn: Button
    private lateinit var resetBtn: Button

    // World points in meters (ARCore world space)
    private val points = mutableListOf<Vector3>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_ar_measure)

        arFragment = supportFragmentManager.findFragmentById(R.id.ar_fragment) as ArFragment
        statusText = findViewById(R.id.status_text)
        doneBtn = findViewById(R.id.done_btn)
        resetBtn = findViewById(R.id.reset_btn)

        // Плоскости оставим включёнными (земля помогает ориентироваться)
        arFragment.arSceneView.planeRenderer.isVisible = true

        // Улучшаем постановку точек на "не плоскостях" (ствол/крона):
        // 1) Instant Placement (можно ставить без найденной плоскости)
        // 2) Depth API (если поддерживается устройством)
        arFragment.arSceneView.session?.let { session ->
            try {
                val config = session.config
                config.instantPlacementMode = Config.InstantPlacementMode.LOCAL_Y_UP
                if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                    config.depthMode = Config.DepthMode.AUTOMATIC
                }
                session.configure(config)
            } catch (_: Throwable) {
                // На некоторых сборках/девайсах может не поддерживаться — работаем без улучшений.
            }
        }

        updateStatus()

        // ✅ Универсальный обработчик тапа:
        // Plane → DepthPoint/Point → InstantPlacementPoint
        // Это важно для деревьев: ствол/крона редко распознаются как плоскость.
        arFragment.arSceneView.scene.setOnTouchListener { _, motionEvent ->
            if (motionEvent.action != MotionEvent.ACTION_UP) return@setOnTouchListener false

            val frame = arFragment.arSceneView.arFrame ?: return@setOnTouchListener false
            if (frame.camera.trackingState != TrackingState.TRACKING) return@setOnTouchListener false

            val bestHit = pickBestHit(frame, motionEvent) ?: return@setOnTouchListener false

            val anchor = bestHit.createAnchor()
            val anchorNode = AnchorNode(anchor)
            anchorNode.setParent(arFragment.arSceneView.scene)

            val pose = anchor.pose
            val p = Vector3(pose.tx(), pose.ty(), pose.tz())
            points.add(p)

            addMarker(anchorNode)
            updateStatus()
            true
        }

        resetBtn.setOnClickListener {
            points.clear()
            val scene = arFragment.arSceneView.scene
            val children = scene.children.toList()
            for (c in children) {
                if (c is AnchorNode) {
                    c.anchor?.detach()
                    scene.removeChild(c)
                }
            }
            updateStatus()
        }

        doneBtn.setOnClickListener {
            val result = buildResultJson()
            val intent = Intent()
            intent.putExtra("result_json", result.toString())
            setResult(Activity.RESULT_OK, intent)
            finish()
        }
    }

    private fun addMarker(parent: AnchorNode) {
        MaterialFactory.makeOpaqueWithColor(this, Color(0.1f, 0.9f, 0.2f))
            .thenAccept { material ->
                // чуть больше, чтобы точно было видно
                val sphere = ShapeFactory.makeSphere(0.03f, Vector3.zero(), material)
                val node = Node()
                node.renderable = sphere
                node.setParent(parent)
                node.localPosition = Vector3.zero()
            }
    }

    // Полная 3D дистанция (на всякий случай)
    private fun dist3d(a: Vector3, b: Vector3): Double {
        val dx = (a.x - b.x).toDouble()
        val dy = (a.y - b.y).toDouble()
        val dz = (a.z - b.z).toDouble()
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    // Для деревьев практичнее:
    // - высота по вертикали (Y)
    // - диаметр/крона по горизонтали (XZ)
    private fun vertical(a: Vector3, b: Vector3): Double {
        return kotlin.math.abs((a.y - b.y).toDouble())
    }

    private fun horizontalXZ(a: Vector3, b: Vector3): Double {
        val dx = (a.x - b.x).toDouble()
        val dz = (a.z - b.z).toDouble()
        return sqrt(dx * dx + dz * dz)
    }

    private fun fmt(m: Double): String = String.format("%.2f м", m)

    private fun updateStatus() {
        val n = points.size

        val height = if (n >= 2) vertical(points[0], points[1]) else null
        val trunk = if (n >= 4) horizontalXZ(points[2], points[3]) else null
        val crown = if (n >= 6) horizontalXZ(points[4], points[5]) else null

        val hint = when (n) {
            0 -> "Точка 1/6: основание ствола (у корня).\nМожно тапать по земле или прямо по стволу."
            1 -> "Точка 2/6: верхушка (видимая точка кроны)"
            2 -> "Точка 3/6: диаметр (1/2) — край ствола (~1.3м)"
            3 -> "Точка 4/6: диаметр (2/2) — другой край ствола"
            4 -> "Точка 5/6: ширина кроны (1/2) — край кроны"
            5 -> "Точка 6/6: ширина кроны (2/2) — другой край кроны"
            else -> "Готово: можно нажать «Сохранить»"
        }

        val lines = StringBuilder()
        lines.append("Высота: ").append(height?.let { fmt(it) } ?: "—").append('\n')
        lines.append("Диаметр ствола: ").append(trunk?.let { fmt(it) } ?: "—").append('\n')
        lines.append("Ширина кроны: ").append(crown?.let { fmt(it) } ?: "—").append('\n')
        lines.append('\n').append(hint)

        statusText.text = lines.toString()
        // Требуем все 6 точек, чтобы пользователь сразу получил 3 метрики одним результатом.
        doneBtn.isEnabled = (n >= 6)
    }

    private fun buildResultJson(): JSONObject {
        val obj = JSONObject()
        obj.put("points_count", points.size)

        if (points.size >= 2) obj.put("height_m", vertical(points[0], points[1]))
        if (points.size >= 4) obj.put("trunk_diameter_m", horizontalXZ(points[2], points[3]))
        if (points.size >= 6) obj.put("crown_width_m", horizontalXZ(points[4], points[5]))

        return obj
    }

    private fun pickBestHit(frame: Frame, motionEvent: MotionEvent): HitResult? {
        val hits = frame.hitTest(motionEvent)

        // Приоритеты:
        // 1) Plane (если есть и подходит) — на земле обычно стабильнее всего
        // 2) DepthPoint / Point — позволяет ставить на дереве без "плоскости"
        // 3) InstantPlacementPoint — последний шанс, но UX сильно лучше, чем "ничего"
        for (hit in hits) {
            val t = hit.trackable
            when (t) {
                is Plane -> {
                    if (t.trackingState == TrackingState.TRACKING && t.isPoseInPolygon(hit.hitPose)) {
                        return hit
                    }
                }
                is DepthPoint -> if (t.trackingState == TrackingState.TRACKING) return hit
                is Point -> if (t.trackingState == TrackingState.TRACKING) return hit
                is InstantPlacementPoint -> if (t.trackingState == TrackingState.TRACKING) return hit
            }
        }

        // Fallback
        return hits.firstOrNull()
    }
}
