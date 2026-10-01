package com.example.arborscan_app

import com.google.ar.core.exceptions.UnavailableApkTooOldException
import com.google.ar.core.exceptions.UnavailableArcoreNotInstalledException
import com.google.ar.core.exceptions.UnavailableDeviceNotCompatibleException
import com.google.ar.core.exceptions.UnavailableException
import com.google.ar.core.exceptions.UnavailableSdkTooOldException
import com.google.ar.core.exceptions.UnavailableUserDeclinedInstallationException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ArAvailabilityFailureTest {
    @Test
    fun declinedInstallationIsCancellationRatherThanAnError() {
        assertNull(ArAvailabilityFailure.message(UnavailableUserDeclinedInstallationException()))
    }

    @Test
    fun serviceAndSdkFailuresHaveDifferentRecoveryInstructions() {
        assertTrue(ArAvailabilityFailure.message(UnavailableArcoreNotInstalledException())!!.contains("установите сервис"))
        assertTrue(ArAvailabilityFailure.message(UnavailableApkTooOldException())!!.contains("обновите сервис"))
        assertEquals("Версия AR в приложении устарела. Обновите ArborScan.",
            ArAvailabilityFailure.message(UnavailableSdkTooOldException()))
        assertTrue(ArAvailabilityFailure.message(UnavailableDeviceNotCompatibleException())!!.contains("по эталону"))
    }

    @Test
    fun failedSessionExplainsRecoveryWithoutExposingInternalExceptionDetails() {
        val exception = UnavailableException("private device diagnostic")
        exception.initCause(IllegalStateException("private camera configuration"))
        val message = ArAvailabilityFailure.message(exception)!!
        assertTrue(message.contains("доступ к камере"))
        assertFalse(message.contains("private"))
        assertFalse(message.contains("IllegalStateException"))
    }
}
