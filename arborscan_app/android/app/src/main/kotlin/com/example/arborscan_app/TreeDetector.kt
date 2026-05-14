package com.example.arborscan_app

import android.graphics.Bitmap

class TreeDetector(private val tflite: TFLiteHelper) {

    data class DetectionResult(
        val mask: Array<Array<FloatArray>>,
        val hasTree: Boolean
    )

    fun detect(bitmap: Bitmap): DetectionResult {

        val mask = tflite.runSegmentation(bitmap)

        val hasTree = checkTreePresence(mask)

        return DetectionResult(mask, hasTree)
    }

    private fun checkTreePresence(mask: Array<Array<FloatArray>>): Boolean {
        var count = 0
        val threshold = 0.5f

        for (y in mask[0].indices) {
            for (x in mask[0][y].indices) {
                if (mask[0][y][x] > threshold) {
                    count++
                }
            }
        }

        return count > 500 // эмпирически
    }
}