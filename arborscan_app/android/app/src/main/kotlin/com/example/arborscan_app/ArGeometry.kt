package com.example.arborscan_app

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pure geometry for ArborScan AR measurement.
 *
 * Y is vertical (ARCore world coordinates). The fixed vertical tree plane is
 * still used for height/DBH-height guidance. DBH itself is NOT measured on
 * that plane: visible trunk edges are tangent rays to an approximately
 * cylindrical trunk, so diameter is recovered from camera-to-axis distance
 * and horizontal angular span.
 */
object ArGeometry {

    private const val EPS = 1e-8

    data class Vec3(
        val x: Double,
        val y: Double,
        val z: Double,
    ) {
        operator fun plus(other: Vec3) = Vec3(x + other.x, y + other.y, z + other.z)
        operator fun minus(other: Vec3) = Vec3(x - other.x, y - other.y, z - other.z)
        operator fun times(scale: Double) = Vec3(x * scale, y * scale, z * scale)

        fun dot(other: Vec3): Double = x * other.x + y * other.y + z * other.z
        fun norm(): Double = sqrt(dot(this))

        fun normalized(): Vec3? {
            val n = norm()
            if (!n.isFinite() || n < EPS) return null
            return this * (1.0 / n)
        }
    }

    data class Ray(
        val origin: Vec3,
        val direction: Vec3,
    )

    data class Plane(
        val point: Vec3,
        val normal: Vec3,
    )

    data class Intersection(
        val point: Vec3,
        val distanceAlongRayM: Double,
        /** 1.0 = perpendicular to plane, 0.0 = parallel. */
        val incidence: Double,
    )

    data class DbhEstimate(
        val diameterM: Double,
        val cameraToAxisM: Double,
        val angularSpanRad: Double,
        val cameraTranslationM: Double,
    )

    fun buildVerticalTreePlane(base: Vec3, cameraAtBaseFix: Vec3): Plane? {
        val horizontal = Vec3(
            cameraAtBaseFix.x - base.x,
            0.0,
            cameraAtBaseFix.z - base.z,
        )
        val normal = horizontal.normalized() ?: return null
        return Plane(point = base, normal = normal)
    }

    fun intersect(ray: Ray, plane: Plane): Intersection? {
        val direction = ray.direction.normalized() ?: return null
        val normal = plane.normal.normalized() ?: return null
        val denom = direction.dot(normal)
        if (!denom.isFinite() || abs(denom) < 1e-4) return null

        val t = (plane.point - ray.origin).dot(normal) / denom
        if (!t.isFinite() || t <= 0.0) return null

        val p = ray.origin + direction * t
        val incidence = abs(direction.dot(normal)).coerceIn(0.0, 1.0)
        return Intersection(point = p, distanceAlongRayM = t, incidence = incidence)
    }

    fun horizontalDistance(a: Vec3, b: Vec3): Double {
        val dx = a.x - b.x
        val dz = a.z - b.z
        return sqrt(dx * dx + dz * dz)
    }

    fun heightAboveBase(base: Vec3, point: Vec3): Double = point.y - base.y

    fun horizontalSpan(a: Vec3, b: Vec3): Double = horizontalDistance(a, b)

    fun relativeHeightError(actualM: Double, targetM: Double): Double = abs(actualM - targetM)

    fun isDbhHeight(
        relativeHeightM: Double,
        targetM: Double = 1.30,
        toleranceM: Double = 0.15,
    ): Boolean {
        if (!relativeHeightM.isFinite()) return false
        return abs(relativeHeightM - targetM) <= toleranceM
    }

    fun midpoint(a: Vec3, b: Vec3): Vec3 = Vec3(
        (a.x + b.x) / 2.0,
        (a.y + b.y) / 2.0,
        (a.z + b.z) / 2.0,
    )

    fun horizontalUnit(direction: Vec3): Vec3? {
        return Vec3(direction.x, 0.0, direction.z).normalized()
    }

    fun horizontalAngle(a: Vec3, b: Vec3): Double? {
        val ua = horizontalUnit(a) ?: return null
        val ub = horizontalUnit(b) ?: return null
        val dot = ua.dot(ub).coerceIn(-1.0, 1.0)
        val angle = acos(dot)
        return if (angle.isFinite()) angle else null
    }

    /**
     * Estimate cylindrical trunk diameter from two tangent sight rays.
     *
     * For a camera at horizontal distance D from the trunk axis and tangent
     * rays separated by angle phi, radius r = D * sin(phi/2).  We require the
     * phone to remain nearly stationary between left/right sightings; when
     * there is a small residual translation the midpoint camera position is
     * used for D and the translation is returned for quality gating.
     */
    fun cylinderDiameterFromTangentRays(
        base: Vec3,
        leftCamera: Vec3,
        leftDirection: Vec3,
        rightCamera: Vec3,
        rightDirection: Vec3,
    ): DbhEstimate? {
        val angle = horizontalAngle(leftDirection, rightDirection) ?: return null
        if (angle <= 1e-5 || angle >= Math.PI / 2.0) return null

        val cameraMid = midpoint(leftCamera, rightCamera)
        val axis = horizontalUnit(base - cameraMid) ?: return null
        val left = horizontalUnit(leftDirection) ?: return null
        val right = horizontalUnit(rightDirection) ?: return null
        // Edges must face and straddle the selected trunk axis. Angular span
        // alone would accept two arbitrary rays pointing away from the tree.
        if (left.dot(axis) <= 0.0 || right.dot(axis) <= 0.0) return null
        val sideLeft = axis.x * left.z - axis.z * left.x
        val sideRight = axis.x * right.z - axis.z * right.x
        if (sideLeft * sideRight > 0.0) return null
        val distance = horizontalDistance(cameraMid, base)
        if (!distance.isFinite() || distance <= 0.0) return null

        val diameter = 2.0 * distance * sin(angle / 2.0)
        if (!diameter.isFinite() || diameter <= 0.0 || diameter >= distance) return null

        return DbhEstimate(
            diameterM = diameter,
            cameraToAxisM = distance,
            angularSpanRad = angle,
            cameraTranslationM = (leftCamera - rightCamera).norm(),
        )
    }

    fun median(values: List<Double>): Double? {
        val clean = values.filter { it.isFinite() }.sorted()
        if (clean.isEmpty()) return null
        val mid = clean.size / 2
        return if (clean.size % 2 == 1) clean[mid] else (clean[mid - 1] + clean[mid]) / 2.0
    }

    fun relativeSpread(values: List<Double>): Double? {
        val clean = values.filter { it.isFinite() && it > 0.0 }
        if (clean.size < 2) return null
        val med = median(clean) ?: return null
        if (med <= EPS) return null
        return (clean.maxOrNull()!! - clean.minOrNull()!!) / med
    }
}
