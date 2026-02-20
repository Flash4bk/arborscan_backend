package com.example.arborscan_app

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import org.json.JSONObject

class MainActivity : FlutterActivity() {

    private val CHANNEL = "arborscan/ar_measure"
    private val REQ_AR_MEASURE = 4242

    private var pendingResult: MethodChannel.Result? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "start" -> {
                        if (pendingResult != null) {
                            result.error("AR_BUSY", "AR measurement is already running", null)
                            return@setMethodCallHandler
                        }

                        pendingResult = result

                        try {
                            val intent = Intent(this, ArMeasureActivity::class.java)
                            val requiredPoints = (call.argument<Int>("required_points") ?: 6)
                            intent.putExtra("required_points", requiredPoints)
                            startActivityForResult(intent, REQ_AR_MEASURE)
                        } catch (e: Exception) {
                            pendingResult = null
                            result.error("AR_START_FAILED", e.toString(), null)
                        }
                    }
                    else -> result.notImplemented()
                }
            }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode != REQ_AR_MEASURE) return

        val res = pendingResult
        pendingResult = null

        if (res == null) return

        if (resultCode != Activity.RESULT_OK || data == null) {
            // cancel
            res.success(null)
            return
        }

        // Пытаемся вытащить разные варианты, чтобы не падать:
        // 1) если ArMeasureActivity положила json строку "result_json"
        val jsonFromActivity = data.getStringExtra("result_json")
        if (!jsonFromActivity.isNullOrBlank()) {
            res.success(jsonFromActivity)
            return
        }

        // 2) если ArMeasureActivity кладёт числа по отдельности
        val meters = data.getDoubleExtra("distanceMeters", Double.NaN)
        val cm = data.getDoubleExtra("distanceCm", Double.NaN)
        val points = data.getIntExtra("points", 0)

        if (meters.isNaN() || cm.isNaN()) {
            // если формат неожиданный — не крэшимся, просто cancel
            res.success(null)
            return
        }

        val obj = JSONObject()
        obj.put("distanceMeters", meters)
        obj.put("distanceCm", cm)
        obj.put("points", points)

        res.success(obj.toString())
    }
}
