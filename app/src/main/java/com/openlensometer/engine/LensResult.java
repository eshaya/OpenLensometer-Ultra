// File: app/src/main/java/com/openlensometer/engine/LensResult.java
package com.openlensometer.engine;

public class LensResult {
    public final float sphere;
    public final float cylinder;
    public final float axis;
    public final float zDistance;

    public LensResult(float sphere, float cylinder, float axis, float zDistance) {
        this.sphere = sphere;
        this.cylinder = cylinder;
        this.axis = axis;
        this.zDistance = zDistance;
    }
}