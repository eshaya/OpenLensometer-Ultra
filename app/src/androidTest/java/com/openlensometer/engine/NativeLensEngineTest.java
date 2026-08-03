// File: NativeLensEngineTest.java
package com.openlensometer.engine;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

@RunWith(AndroidJUnit4.class)
public class NativeLensEngineTest {

    @Test
    public void nativeLibraryLoadsAndResponds() {
        NativeLensEngine engine = new NativeLensEngine();
        String response = engine.pingEngine();
        assertNotNull("The JNI bridge returned a null string.", response);
        assertEquals("JNI Bridge Active: C++ Math Engine Ready", response);
    }
}
