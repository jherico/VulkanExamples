/*
* Basic camera class
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "common.hpp"

class Camera {
private:
    const float MAX_PITCH{ (float)M_PI_2 * 0.95f };
    float fov{ 60.0f };
    float znear{ 0.11f }, zfar{ 512.0f };
    float aspect{ 1.0f };

    void updateViewMatrix() {
        // Constrain pitch to ~-PI/2 to ~PI/2
        if (abs(yawPitch.y) > MAX_PITCH) {
            yawPitch.y = std::max(std::min(yawPitch.y, MAX_PITCH), -MAX_PITCH);
        }
        while (abs(yawPitch.x) > M_PI) {
            yawPitch.x += (float)((yawPitch.x > 0) ? -M_PI : M_PI);
        }

        if (type == CameraType::firstperson) {
            const_cast<glm::quat&>(orientation) = glm::angleAxis(yawPitch.y, Vectors::RIGHT) * glm::angleAxis(yawPitch.x, Vectors::UP);
            glm::mat4 rotM = glm::mat4_cast(orientation);
            glm::mat4 transM = glm::translate(glm::mat4(), position);
            matrices.view = rotM * transM;
        } else {
            const_cast<glm::quat&>(orientation) = glm::angleAxis(yawPitch.x, Vectors::UP) * glm::angleAxis(yawPitch.y, Vectors::RIGHT);
            // rotate into the camera xy plane
            glm::vec3 origin = orientation * glm::vec3(position.x, position.y, 0.0f);
            glm::vec3 cameraPosition = orientation * glm::vec3(0, 0, -position.z);
            matrices.view = glm::lookAt(cameraPosition + origin, origin, glm::vec3(0, 1, 0));
        }
        matrices.skyboxView = matrices.view;
        matrices.skyboxView[3] = glm::vec4(0, 0, 0, 1);
    }

public:
    enum CameraType
    {
        lookat,
        firstperson
    };
    CameraType type{ CameraType::lookat };

    const glm::quat orientation;
    glm::vec2 yawPitch;
    glm::vec3 position;

    float rotationSpeed{ 1.0f };
    float movementSpeed{ 1.0f };

    struct Matrices {
        glm::mat4 perspective;
        glm::mat4 view;
        glm::mat4 skyboxView;
    } matrices;

    struct Keys {
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
    } keys;

    Camera() { matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar); }

    bool moving() { return keys.left || keys.right || keys.up || keys.down; }

    void setFieldOfView(float fov) {
        this->fov = fov;
        matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar);
    }

    void setAspectRatio(const glm::vec2& size) { setAspectRatio(size.x / size.y); }

    void setAspectRatio(const vk::Extent2D& size) { setAspectRatio((float)size.width / (float)size.height); }

    void setAspectRatio(float aspect) {
        this->aspect = aspect;
        matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar);
    }

    void setNear(float newZnear) { setPerspective(fov, aspect, newZnear, zfar); }

    void setFar(float newZfar) { setPerspective(fov, aspect, znear, newZfar); }

    void setPerspective(float fov, const glm::vec2& size, float znear = 0.1f, float zfar = 512.0f) { setPerspective(fov, size.x / size.y, znear, zfar); }

    void setPerspective(float fov, const vk::Extent2D& size, float znear = 0.1f, float zfar = 512.0f) {
        setPerspective(fov, (float)size.width / (float)size.height, znear, zfar);
    }

    void setPerspective(float fov, float aspect, float znear = 0.1f, float zfar = 512.0f) {
        this->aspect = aspect;
        this->fov = fov;
        this->znear = znear;
        this->zfar = zfar;
        matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar);
    };

    void setRotation(const glm::vec3& rotation) {
        yawPitch = glm::radians(glm::vec2(-rotation.y, -rotation.x));
        updateViewMatrix();
    };

    void setPosition(const glm::vec3& translation) { setTranslation(translation); }

    void setTranslation(const glm::vec3& translation) {
        position = vec3(-translation.x, -translation.y, translation.z);
        updateViewMatrix();
    }

    void setZoom(float f) { setTranslation({ 0, 0, f }); }

    void rotate(const glm::vec2& delta) {
        yawPitch += delta;
        updateViewMatrix();
    }

    void rotate(const glm::vec3& delta) { yawPitch += glm::radians(glm::vec2(delta)); }

    // Translate in the Z axis of the camera
    void dolly(float delta) { translate(glm::vec3(0, 0, delta)); }

    // Translate in the XY plane of the camera
    void translate(const glm::vec2& delta) { translate(glm::vec3(delta.x, delta.y, 0)); }

    void translate(const glm::vec3& delta) {
        if (type == CameraType::firstperson) {
            position += glm::inverse(orientation) * delta;
        } else {
            position += delta;
        }
        updateViewMatrix();
    }

    void keyPressed(uint32_t key, uint32_t mods) {}

    void keyReleased(uint32_t key, uint32_t mods) {}

    void update(float deltaTime) {
        if (type == CameraType::firstperson) {
            if (moving()) {
                glm::vec3 camFront = orientation * glm::vec3{ 0, 0, 1 };
                float moveSpeed = deltaTime * movementSpeed;

                if (keys.up)
                    position += camFront * moveSpeed;
                if (keys.down)
                    position -= camFront * moveSpeed;
                if (keys.left)
                    position -= glm::normalize(glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f))) * moveSpeed;
                if (keys.right)
                    position += glm::normalize(glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f))) * moveSpeed;

                updateViewMatrix();
            }
        }
    }

    // Update camera passing separate axis data (gamepad)
    // Returns true if view or position has been changed
    bool updatePad(const glm::vec2& axisLeft, const glm::vec2& axisRight, float deltaTime) {
        bool retVal = false;

        if (type == CameraType::firstperson) {
            // Use the common console thumbstick layout
            // Left = view, right = move

            const float deadZone = 0.0015f;
            const float range = 1.0f - deadZone;
            glm::vec3 camFront = orientation * glm::vec3{ 0, 0, 1 };

            float moveSpeed = deltaTime * movementSpeed * 2.0f;
            float rotSpeed = deltaTime * 50.0f;

            // Move
            if (fabsf(axisLeft.y) > deadZone) {
                float pos = (fabsf(axisLeft.y) - deadZone) / range;
                position -= camFront * pos * ((axisLeft.y < 0.0f) ? -1.0f : 1.0f) * moveSpeed;
                retVal = true;
            }
            if (fabsf(axisLeft.x) > deadZone) {
                float pos = (fabsf(axisLeft.x) - deadZone) / range;
                position += glm::normalize(glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f))) * pos * ((axisLeft.x < 0.0f) ? -1.0f : 1.0f) * moveSpeed;
                retVal = true;
            }

            // Rotate
            if (fabsf(axisRight.x) > deadZone) {
                float pos = (fabsf(axisRight.x) - deadZone) / range;
                yawPitch.x += glm::radians(pos * ((axisRight.x < 0.0f) ? -1.0f : 1.0f) * rotSpeed);
                retVal = true;
            }

            if (fabsf(axisRight.y) > deadZone) {
                float pos = (fabsf(axisRight.y) - deadZone) / range;
                yawPitch.y *= glm::radians(pos * ((axisRight.x < 0.0f) ? -1.0f : 1.0f) * rotSpeed);
                retVal = true;
            }
        } else {
            // todo: move code from example base class for look-at
        }

        if (retVal) {
            updateViewMatrix();
        }

        return retVal;
    }
};
