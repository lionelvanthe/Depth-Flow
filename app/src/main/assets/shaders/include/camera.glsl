#ifndef SHADERFLOW_CAMERA
#define SHADERFLOW_CAMERA

    // Camera Mode Enum
    const int CameraModeFreeCamera = 0;
    const int CameraMode2D         = 1;
    const int CameraModeSpherical  = 2;

    // Camera Projection Enum
    const int CameraProjectionPerspective     = 0;
    const int CameraProjectionStereoscopic    = 1;
    const int CameraProjectionEquirectangular = 2;

    struct Camera {
        int mode;
        int projection;
        vec3 position;
        vec3 up;
        vec3 down;
        vec3 left;
        vec3 right;
        vec3 forward;
        vec3 backward;
        vec3 zenith;

        //// Rays 3D

        vec3 origin;   // Origin of the camera ray
        vec3 target;   // Target of the camera ray
        float orbital; // Displaces ray origin and ray target backwards
        float dolly;   // Displaces ray origin backwards

        //// Rays 2D

        vec3 plane_point;
        vec3 plane_normal;
        vec2 gluv;
        vec2 agluv;
        vec2 stuv;
        vec2 astuv;
        vec2 glxy;
        vec2 stxy;
        bool out_of_bounds;

        //// Virtual Reality
        float separation;

        //// Parameters
        float focal_length;
        float isometric;
        float zoom;
    };

    /* Build a projection plane at the origin where the camera is looking */
    vec3 CameraRectangle(Camera camera, vec2 gluv, float size) {
        return size*(gluv.x*camera.right + gluv.y*camera.up);
    }

    vec3 CameraRayOrigin(Camera camera, vec2 gluv) {
        return camera.position
            + CameraRectangle(camera, gluv, camera.zoom*camera.isometric)
            + (camera.backward*camera.orbital)
            + (camera.backward*camera.dolly);
    }

    vec3 CameraRayTarget(Camera camera, vec2 gluv) {
        return camera.position
            + CameraRectangle(camera, gluv, camera.zoom)
            + (camera.backward*camera.orbital)
            + (camera.forward *camera.focal_length);
    }

    Camera CameraRay2D(Camera camera) {

        // Calculate the interstion with the plane define by a point and norm
        // Note: (t < 0) the intersection is behind the camera
        // https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
        float num = dot(camera.plane_point - camera.origin, camera.plane_normal);
        float den = dot(camera.target - camera.origin, camera.plane_normal);
        float t = (num/den);

        // Calculate the intersection point
        camera.out_of_bounds = (t < 0.0) || (abs(gluv.x) > iWantAspect + 0.1);
        camera.gluv  = (camera.origin + (t*(camera.target - camera.origin))).xy;
        camera.agluv = (camera.gluv / vec2(iAspectRatio, 1.0));
        camera.stuv  = (camera.gluv  + 1.0)/2.0;
        camera.astuv = (camera.agluv + 1.0)/2.0;
        camera.stxy  = (iResolution * camera.astuv);
        camera.glxy  = (camera.stxy - iResolution/2.0);
        return camera;
    }

    Camera CameraProject(Camera camera) {

        // Simple origin and target
        if (camera.projection == CameraProjectionPerspective) {
            camera.origin = CameraRayOrigin(camera, gluv);
            camera.target = CameraRayTarget(camera, gluv);

        // Emulate two cameras, same as perspective
        } else if (camera.projection == CameraProjectionStereoscopic) {

            // Each side of the screen has its own gluv at the center
            vec2 gluv = gluv - sign(agluv.x) * vec2(iAspectRatio/2.0, 0.0);

            // The eyes are two cameras displaced by the separation
            camera.position += (sign(agluv.x) * camera.separation) * camera.right;
            camera.origin = CameraRayOrigin(camera, gluv);
            camera.target = CameraRayTarget(camera, gluv);

        // Map the screen rectangle to azimuth and inclination
        } else if (camera.projection == CameraProjectionEquirectangular) {

            // Map a sphere to the screen,
            float inclination = (camera.zoom) * (PI*agluv.y/2.0);
            float azimuth     = (camera.zoom) * (PI*agluv.x/1.0);

            // Rotate the forward vector
            vec3 target = camera.forward;
            target = rotate3d(target, camera.right,  -inclination);
            target = rotate3d(target, camera.up, +azimuth);

            // All rays originate from the position
            camera.origin = camera.position;
            camera.target = camera.position + target;
        }

        // Expect ray marching scenes to normalize it
        return CameraRay2D(camera);
    }

    #define GetCamera(name) \
        Camera name; \
        { \
            name.plane_point   = vec3(0.0, 0.0, 1.0); \
            name.plane_normal  = vec3(0.0, 0.0, 1.0); \
            name.mode          = iCameraMode; \
            name.projection    = iCameraProjection; \
            name.position      = iCameraPosition; \
            name.orbital       = iCameraOrbital; \
            name.dolly         = iCameraDolly; \
            name.zenith        = iCameraZenith; \
            name.up            = iCameraUpward; \
            name.down          = iCameraUpward*(-1.0); \
            name.left          = iCameraRight*(-1.0); \
            name.right         = iCameraRight; \
            name.forward       = iCameraForward; \
            name.backward      = iCameraForward*(-1.0); \
            name.isometric     = iCameraIsometric; \
            name.focal_length  = iCameraFocalLength; \
            name.zoom          = iCameraZoom; \
            name.separation    = iCameraSeparation; \
            name.out_of_bounds = false; \
            name = CameraProject(name); \
        }

#endif
