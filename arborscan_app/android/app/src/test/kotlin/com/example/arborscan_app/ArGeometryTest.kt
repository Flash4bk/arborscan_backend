package com.example.arborscan_app
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.asin
import kotlin.math.sin
import kotlin.math.cos

class ArGeometryTest {
    private val base = ArGeometry.Vec3(0.0, 0.0, 0.0)
    private val camera = ArGeometry.Vec3(0.0, 1.3, 3.0)
    @Test fun tangentRaysRecoverKnownCylinderAndRejectWrongTree() {
        val a = asin(0.15 / 3.0)
        val left = ArGeometry.Vec3(-sin(a), 0.0, -cos(a))
        val right = ArGeometry.Vec3(sin(a), 0.0, -cos(a))
        val estimate = ArGeometry.cylinderDiameterFromTangentRays(base,camera,left,camera,right)!!
        assertEquals(0.30,estimate.diameterM,1e-10)
        assertNull(ArGeometry.cylinderDiameterFromTangentRays(base,camera,left * -1.0,camera,right * -1.0))
        assertNull(ArGeometry.cylinderDiameterFromTangentRays(base,camera,left,camera,ArGeometry.Vec3(-0.2,0.0,-1.0)))
        val moved = ArGeometry.cylinderDiameterFromTangentRays(base,camera,left,camera+ArGeometry.Vec3(0.0,0.2,0.0),right)!!
        assertEquals(0.2,moved.cameraTranslationM,1e-10)
    }
    @Test fun heightUsesWorldVerticalAndRejectsParallelOrBehindCamera() {
        val plane = ArGeometry.buildVerticalTreePlane(base,camera)!!
        val top = ArGeometry.intersect(ArGeometry.Ray(camera,ArGeometry.Vec3(0.0,8.7,-3.0)),plane)!!
        assertEquals(10.0,ArGeometry.heightAboveBase(base,top.point),1e-10)
        assertNull(ArGeometry.intersect(ArGeometry.Ray(camera,ArGeometry.Vec3(1.0,0.0,0.0)),plane))
        assertNull(ArGeometry.intersect(ArGeometry.Ray(camera,ArGeometry.Vec3(0.0,0.0,1.0)),plane))
        assertNull(ArGeometry.buildVerticalTreePlane(base,ArGeometry.Vec3(0.0,2.0,0.0)))
    }
}
