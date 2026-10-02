package com.openlensometer.engine

import android.content.Context
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.util.UUID

data class EyePrescription(
    val sphere: Double,
    val cylinder: Double,
    val axis: Int,
    val add: Double
)

data class ReferenceGlasses(
    val id: String,
    val name: String,
    val rightEye: EyePrescription,
    val leftEye: EyePrescription
)

class ReferenceGlassesStore(context: Context) {
    private val preferences = context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)

    fun all(): List<ReferenceGlasses> = readAll()

    fun active(): ReferenceGlasses? {
        val activeId = preferences.getString(KEY_ACTIVE_ID, null) ?: return null
        return readAll().firstOrNull { it.id == activeId }
    }

    fun select(id: String) {
        require(readAll().any { it.id == id }) { "Unknown reference glasses" }
        preferences.edit().putString(KEY_ACTIVE_ID, id).apply()
    }

    fun add(
        name: String,
        rightEye: EyePrescription,
        leftEye: EyePrescription
    ): ReferenceGlasses {
        val normalizedName = name.trim()
        require(normalizedName.isNotEmpty()) { "Enter a name for these glasses" }
        validate(rightEye)
        validate(leftEye)

        val entry = ReferenceGlasses(UUID.randomUUID().toString(), normalizedName, rightEye, leftEye)
        writeAll(readAll() + entry)
        preferences.edit().putString(KEY_ACTIVE_ID, entry.id).apply()
        return entry
    }

    fun removeActive(): Boolean {
        val selected = active() ?: return false
        val remaining = readAll().filterNot { it.id == selected.id }
        writeAll(remaining)
        preferences.edit()
            .putString(KEY_ACTIVE_ID, remaining.firstOrNull()?.id)
            .apply()
        return true
    }

    private fun validate(prescription: EyePrescription) {
        require(prescription.sphere.isFinite() && prescription.sphere in -30.0..30.0)
        require(prescription.cylinder.isFinite() && prescription.cylinder in -20.0..20.0)
        require(prescription.axis in 0..180)
        require(prescription.add.isFinite() && prescription.add in 0.0..10.0)
    }

    private fun readAll(): List<ReferenceGlasses> {
        val json = preferences.getString(KEY_RECORDS, "[]") ?: "[]"
        return try {
            val array = JSONArray(json)
            buildList(array.length()) {
                for (index in 0 until array.length()) {
                    val item = array.optJSONObject(index) ?: continue
                    try {
                        add(item.toReferenceGlasses())
                    } catch (_: JSONException) {
                        continue
                    } catch (_: IllegalArgumentException) {
                        continue
                    }
                }
            }
        } catch (_: JSONException) {
            emptyList()
        }
    }

    private fun writeAll(entries: List<ReferenceGlasses>) {
        val array = JSONArray()
        entries.forEach { entry -> array.put(entry.toJson()) }
        preferences.edit().putString(KEY_RECORDS, array.toString()).apply()
    }

    private fun ReferenceGlasses.toJson() = JSONObject()
        .put("id", id)
        .put("name", name)
        .put("right", rightEye.toJson())
        .put("left", leftEye.toJson())

    private fun EyePrescription.toJson() = JSONObject()
        .put("sphere", sphere)
        .put("cylinder", cylinder)
        .put("axis", axis)
        .put("add", add)

    private fun JSONObject.toReferenceGlasses() = ReferenceGlasses(
        getString("id"),
        getString("name"),
        getJSONObject("right").toEyePrescription(),
        getJSONObject("left").toEyePrescription()
    )

    private fun JSONObject.toEyePrescription() = EyePrescription(
        getDouble("sphere"),
        getDouble("cylinder"),
        getInt("axis"),
        getDouble("add")
    )

    private companion object {
        const val PREFERENCES_NAME = "reference_glasses"
        const val KEY_RECORDS = "records"
        const val KEY_ACTIVE_ID = "active_id"
    }
}