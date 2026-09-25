package com.example.arborscan_app

import android.app.Activity
import android.content.ClipData
import android.content.Intent
import androidx.core.content.FileProvider
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File

/** Scoped PDF handoff. No storage permissions, broad provider paths or arbitrary files. */
class ReportExport(private val activity: Activity) {
    companion object { const val REQUEST_SAVE = 1002 }
    private var pending: MethodChannel.Result? = null
    private var source: File? = null
    fun handle(call: MethodCall, result: MethodChannel.Result) {
        if (call.method !in listOf("save", "open", "share")) { result.notImplemented(); return }
        if (pending != null) { result.error("busy", "PDF operation already running", null); return }
        try {
            val root = File(activity.cacheDir, "report-export").canonicalFile
            val file = File(call.argument<String>("path") ?: "").canonicalFile
            require(file.parentFile == root && file.extension == "pdf" && file.isFile)
            val name = call.argument<String>("name") ?: "ArborScan.pdf"
            require(name.matches(Regex("[A-Za-z0-9_-]+\\.pdf")))
            if (call.method == "save") {
                pending = result; source = file
                activity.startActivityForResult(Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE); type = "application/pdf"
                    putExtra(Intent.EXTRA_TITLE, name)
                }, REQUEST_SAVE)
            } else {
                val uri = FileProvider.getUriForFile(activity, "${activity.packageName}.reportfiles", file)
                val intent = Intent(if (call.method == "share") Intent.ACTION_SEND else Intent.ACTION_VIEW).apply {
                    if (call.method == "share") { type = "application/pdf"; putExtra(Intent.EXTRA_STREAM, uri) }
                    else setDataAndType(uri, "application/pdf")
                    clipData = ClipData.newRawUri("PDF", uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                activity.startActivity(if (call.method == "share") Intent.createChooser(intent, "Поделиться PDF") else intent)
                result.success("opened")
            }
        } catch (_: Exception) {
            pending = null; source = null
            result.error("pdf_action_failed", "PDF action unavailable", null)
        }
    }
    fun onResult(code: Int, resultCode: Int, data: Intent?): Boolean {
        if (code != REQUEST_SAVE) return false
        val callback = pending ?: return true
        val file = source
        if (resultCode != Activity.RESULT_OK || data?.data == null) {
            pending = null; source = null; callback.success("cancelled"); return true
        }
        val uri = data.data!!
        Thread {
            val ok = try {
                requireNotNull(file)
                activity.contentResolver.openOutputStream(uri, "wt").use { out ->
                    requireNotNull(out); file.inputStream().use { it.copyTo(out) }; out.flush()
                }; true
            } catch (_: Exception) { false }
            activity.runOnUiThread {
                pending = null; source = null
                if (ok) callback.success("saved") else callback.error("write_failed", "Could not save PDF", null)
            }
        }.start()
        return true
    }
}
