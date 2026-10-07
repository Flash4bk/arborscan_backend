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
    fun unsupportedOrUnknownGraphicsExplainAlternativeBeforeSceneformStarts() {
        for (version in listOf(0, 0x10000, 0x20000)) {
            val message = ArAvailabilityFailure.graphicsMessage(version)!!
            assertTrue(message.contains("не поддерживает AR"))
            assertTrue(message.contains("по эталону"))
        }
    }

    @Test
    fun supportedGraphicsStillReachNormalArServiceHandling() {
        assertNull(ArAvailabilityFailure.graphicsMessage(0x30000))
        assertNull(ArAvailabilityFailure.graphicsMessage(0x30001))
    }

    @Test
    fun wrappedRendererFailureExplainsAlternativeWithoutDriverDetails() {
        val exception = RuntimeException("private layout", IllegalStateException("Couldn't create Engine: private driver"))
        val message = ArAvailabilityFailure.initializationMessage(exception)!!
        assertTrue(message.contains("графику AR"))
        assertTrue(message.contains("по эталону"))
        assertFalse(message.contains("private"))
        assertFalse(message.contains("Engine"))
    }

    @Test
    fun wrappedSessionFailurePreservesRecoveryAndDeclinedCancellation() {
        val outdated = RuntimeException("layout", RuntimeException("session", UnavailableApkTooOldException()))
        assertEquals(ArAvailabilityFailure.message(UnavailableApkTooOldException()),
            ArAvailabilityFailure.initializationMessage(outdated))
        val declined = RuntimeException("layout", UnavailableUserDeclinedInstallationException())
        assertNull(ArAvailabilityFailure.initializationMessage(declined))
    }

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
