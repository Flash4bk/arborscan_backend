package com.example.arborscan_app

import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException

/** A declined installer is cancellation; other session failures need a visible explanation. */
internal object ArAvailabilityFailure {
    // Sceneform requires OpenGL ES 3.0. Reject unsupported or unknown graphics
    // before layout inflation, while ordinary photo/reference analysis remains usable.
    fun graphicsMessage(requiredGlEsVersion: Int): String? =
        if (requiredGlEsVersion >= 0x30000) null
        else "Графика этого устройства не поддерживает AR. Можно продолжить анализ фото или измерение по эталону."

    // Layout inflation wraps renderer or session failures. Preserve meaningful
    // ARCore recovery/cancellation without exposing private driver diagnostics.
    fun initializationMessage(exception: Exception): String? {
        var cause: Throwable? = exception
        repeat(8) {
            val current = cause
            if (current is UnavailableException) return message(current)
            cause = current?.cause
        }
        return "Не удалось запустить графику AR на этом устройстве. Можно продолжить анализ фото или измерение по эталону."
    }

    fun message(exception: UnavailableException): String? = when (exception) {
        is UnavailableUserDeclinedInstallationException -> null
        is UnavailableArcoreNotInstalledException ->
            "Для AR установите сервис Google Play Services for AR и повторите запуск."
        is UnavailableApkTooOldException ->
            "Для AR обновите сервис Google Play Services for AR и повторите запуск."
        is UnavailableSdkTooOldException ->
            "Версия AR в приложении устарела. Обновите ArborScan."
        is UnavailableDeviceNotCompatibleException ->
            "AR недоступен на этом устройстве. Можно продолжить анализ фото или измерение по эталону."
        else ->
            "Не удалось запустить AR-сессию. Проверьте доступ к камере и сервис Google Play Services for AR, затем повторите запуск."
    }
}
