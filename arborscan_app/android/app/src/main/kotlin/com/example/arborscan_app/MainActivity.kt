package com.example.arborscan_app

import android.app.Activity
import android.content.Intent
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {

    private val channelName = "arborscan/ar_measure"
    private var pendingResult: MethodChannel.Result? = null
    private val requestCodeAr = 1001

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, channelName)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "start" -> {
                        if (pendingResult != null) {
                            result.error("busy", "AR measurement is already running", null)
                            return@setMethodCallHandler
                        }

                        pendingResult = result
                        val intent = Intent(this, ArMeasureActivity::class.java)
                        startActivityForResult(intent, requestCodeAr)
                    }

                    else -> result.notImplemented()
                }
            }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode != requestCodeAr) return

        val callback = pendingResult
        pendingResult = null

        if (callback == null) return

        if (resultCode == Activity.RESULT_OK) {
            callback.success(data?.getStringExtra(ArMeasureActivity.EXTRA_RESULT_JSON))
        } else {
            callback.success(null)
        }
    }
}
