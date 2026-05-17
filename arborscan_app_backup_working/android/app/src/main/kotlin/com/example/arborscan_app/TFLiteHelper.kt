package com.example.arborscan_app

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class TFLiteHelper(context: Context) {

    private val interpreter: Interpreter

    init {
        val asset = context.assets.open("best_float16.tflite")
        val model = asset.readBytes()

        val buffer = ByteBuffer.allocateDirect(model.size)
        buffer.order(ByteOrder.nativeOrder())
        buffer.put(model)

        interpreter = Interpreter(buffer)
    }

    fun runSegmentation(bitmap: Bitmap): Array<Array<FloatArray>> {

        val input = preprocess(bitmap)

        val output = Array(1) { Array(160) { FloatArray(160) } }

        interpreter.run(input, output)

        return output
    }

    private fun preprocess(bitmap: Bitmap): ByteBuffer {
        val inputSize = 640
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        val buffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        buffer.order(ByteOrder.nativeOrder())

        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val pixel = resized.getPixel(x, y)

                buffer.putFloat(((pixel shr 16 and 0xFF) / 255f))
                buffer.putFloat(((pixel shr 8 and 0xFF) / 255f))
                buffer.putFloat(((pixel and 0xFF) / 255f))
            }
        }

        return buffer
    }
}