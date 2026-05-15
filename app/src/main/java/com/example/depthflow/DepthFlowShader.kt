package com.example.depthflow

object DepthFlowShader {

    val VERTEX_SHADER = """
        #version 300 es
        layout(location = 0) in vec4 aPosition;
        layout(location = 1) in vec2 aTexCoord;
        out vec2 vTexCoord;
        void main() {
            gl_Position = aPosition;
            vTexCoord = aTexCoord;
        }
    """.trimIndent()

    fun getFragmentShader(mainLogic: String): String {
        val header = """
            #version 300 es
            precision highp float;
            precision highp sampler2D;

            in vec2 vTexCoord;
            out vec4 fragColor;

            uniform sampler2D image;
            uniform sampler2D depth;

            // State Uniforms
            uniform float iVigIntensity;
            uniform float iVigDecay;
            uniform float iLensIntensity;
            uniform float iLensDecay;
            uniform int iLensQuality;
            uniform float iInpaint;
            uniform float iBlurIntensity;
            uniform float iBlurStart;
            uniform float iBlurEnd;
            uniform float iBlurExponent;
            uniform int iBlurQuality;
            uniform int iBlurDirections;
            uniform float iColorsSaturation;
            uniform float iColorsContrast;
            uniform float iColorsBrightness;
            uniform float iColorsSepia;
            uniform float iDepthHeight;
            uniform float iDepthSteady;
            uniform float iDepthFocus;
            uniform float iDepthZoom;
            uniform float iDepthIsometric;
            uniform float iDepthDolly;
            uniform vec2 iDepthOffset;
            uniform vec2 iDepthCenter;
            uniform vec2 iDepthOrigin;
            
            uniform float iQuality;
            uniform float iTime;

            #define TAU 6.283185307179586
            #define agluv (vTexCoord * 2.0 - 1.0)
            #define astuv vTexCoord

            struct Camera {
                vec3 position;
                float zoom;
                float isometric;
                float dolly;
                float focal_length;
                vec3 plane_point;
                vec2 gluv;
                bool out_of_bounds;
                vec3 origin;
            };

            Camera CameraProject(Camera c) {
                // Simplified projection logic for mobile
                c.gluv = vTexCoord;
                c.out_of_bounds = false;
                c.origin = vec3(vTexCoord, 0.0);
                return c;
            }

            vec4 gtexture(sampler2D s, vec2 uv, bool clamp) {
                if (clamp) {
                    uv = clamp(uv, 0.0, 1.0);
                }
                return texture(s, uv);
            }

            float angle(vec3 a, vec3 b) {
                return acos(dot(normalize(a), normalize(b)));
            }

            vec3 rgb2hsv(vec3 c) {
                vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                float d = q.x - min(q.w, q.y);
                float e = 1.0e-10;
                return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
            }

            vec3 hsv2rgb(vec3 c) {
                vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
            }

            #define GetCamera(name) \
                Camera name; \
                name.position = vec3(0.0); \
                name.zoom = 1.0; \
                name.isometric = 0.0; \
                name.dolly = 0.0; \
                name.focal_length = 1.0; \
                name.plane_point = vec3(0.0, 0.0, 1.0); \
                name.gluv = vTexCoord; \
                name.out_of_bounds = false; \
                name.origin = vec3(vTexCoord, 0.0);
        """.trimIndent()
        
        return header + "\n" + mainLogic
    }
}
