/*
* Vulkan Example - Animated gears using multiple uniform buffers
*
* See readme.md for details
*
* Copyright (C) 2015 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkanGear.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

int32_t VulkanGear::newVertex(std::vector<Vertex>* vBuffer, float x, float y, float z, const glm::vec3& normal) {
    Vertex v(glm::vec3(x, y, z), normal, color);
    vBuffer->push_back(v);
    return (uint32_t)vBuffer->size() - 1;
}

void VulkanGear::newFace(std::vector<uint32_t>* iBuffer, int a, int b, int c) {
    iBuffer->push_back(a);
    iBuffer->push_back(b);
    iBuffer->push_back(c);
}

VulkanGear::~VulkanGear() {
    // Clean up vulkan resources
    uniformData.destroy();
    meshInfo.destroy();
}

void VulkanGear::generate(const vks::Context& context,
                          float inner_radius,
                          float outer_radius,
                          float width,
                          int teeth,
                          float tooth_depth,
                          glm::vec3 color,
                          glm::vec3 pos,
                          float rotSpeed,
                          float rotOffset) {
    device = context.device;
    this->color = color;
    this->pos = pos;
    this->rotOffset = rotOffset;
    this->rotSpeed = rotSpeed;

    std::vector<Vertex> vBuffer;
    std::vector<uint32_t> iBuffer;

    int i;
    float r0, r1, r2;
    float ta, da;
    float u1, v1, u2, v2, len;
    float cos_ta, cos_ta_1da, cos_ta_2da, cos_ta_3da, cos_ta_4da;
    float sin_ta, sin_ta_1da, sin_ta_2da, sin_ta_3da, sin_ta_4da;
    int32_t ix0, ix1, ix2, ix3, ix4, ix5;

    r0 = inner_radius;
    r1 = outer_radius - tooth_depth / 2.0f;
    r2 = outer_radius + tooth_depth / 2.0f;
    da = 2.0f * (float)M_PI / teeth / 4.0f;

    glm::vec3 normal;

    for (i = 0; i < teeth; i++) {
        ta = i * 2.0f * (float)M_PI / teeth;
        // todo : naming
        cos_ta = cos(ta);
        cos_ta_1da = cos(ta + da);
        cos_ta_2da = cos(ta + 2 * da);
        cos_ta_3da = cos(ta + 3 * da);
        cos_ta_4da = cos(ta + 4 * da);
        sin_ta = sin(ta);
        sin_ta_1da = sin(ta + da);
        sin_ta_2da = sin(ta + 2 * da);
        sin_ta_3da = sin(ta + 3 * da);
        sin_ta_4da = sin(ta + 4 * da);

        u1 = r2 * cos_ta_1da - r1 * cos_ta;
        v1 = r2 * sin_ta_1da - r1 * sin_ta;
        len = sqrt(u1 * u1 + v1 * v1);
        u1 /= len;
        v1 /= len;
        u2 = r1 * cos_ta_3da - r2 * cos_ta_2da;
        v2 = r1 * sin_ta_3da - r2 * sin_ta_2da;

        // front face
        normal = glm::vec3(0.0, 0.0, 1.0);
        ix0 = newVertex(&vBuffer, r0 * cos_ta, r0 * sin_ta, width * 0.5f, normal);
        ix1 = newVertex(&vBuffer, r1 * cos_ta, r1 * sin_ta, width * 0.5f, normal);
        ix2 = newVertex(&vBuffer, r0 * cos_ta, r0 * sin_ta, width * 0.5f, normal);
        ix3 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, width * 0.5f, normal);
        ix4 = newVertex(&vBuffer, r0 * cos_ta_4da, r0 * sin_ta_4da, width * 0.5f, normal);
        ix5 = newVertex(&vBuffer, r1 * cos_ta_4da, r1 * sin_ta_4da, width * 0.5f, normal);
        newFace(&iBuffer, ix0, ix1, ix2);
        newFace(&iBuffer, ix1, ix3, ix2);
        newFace(&iBuffer, ix2, ix3, ix4);
        newFace(&iBuffer, ix3, ix5, ix4);

        // front sides of teeth
        normal = glm::vec3(0.0, 0.0, 1.0);
        ix0 = newVertex(&vBuffer, r1 * cos_ta, r1 * sin_ta, width * 0.5f, normal);
        ix1 = newVertex(&vBuffer, r2 * cos_ta_1da, r2 * sin_ta_1da, width * 0.5f, normal);
        ix2 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, width * 0.5f, normal);
        ix3 = newVertex(&vBuffer, r2 * cos_ta_2da, r2 * sin_ta_2da, width * 0.5f, normal);
        newFace(&iBuffer, ix0, ix1, ix2);
        newFace(&iBuffer, ix1, ix3, ix2);

        // back face
        normal = glm::vec3(0.0, 0.0, -1.0);
        ix0 = newVertex(&vBuffer, r1 * cos_ta, r1 * sin_ta, -width * 0.5f, normal);
        ix1 = newVertex(&vBuffer, r0 * cos_ta, r0 * sin_ta, -width * 0.5f, normal);
        ix2 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, -width * 0.5f, normal);
        ix3 = newVertex(&vBuffer, r0 * cos_ta, r0 * sin_ta, -width * 0.5f, normal);
        ix4 = newVertex(&vBuffer, r1 * cos_ta_4da, r1 * sin_ta_4da, -width * 0.5f, normal);
        ix5 = newVertex(&vBuffer, r0 * cos_ta_4da, r0 * sin_ta_4da, -width * 0.5f, normal);
        newFace(&iBuffer, ix0, ix1, ix2);
        newFace(&iBuffer, ix1, ix3, ix2);
        newFace(&iBuffer, ix2, ix3, ix4);
        newFace(&iBuffer, ix3, ix5, ix4);

        // back sides of teeth
        normal = glm::vec3(0.0, 0.0, -1.0);
        ix0 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, -width * 0.5f, normal);
        ix1 = newVertex(&vBuffer, r2 * cos_ta_2da, r2 * sin_ta_2da, -width * 0.5f, normal);
        ix2 = newVertex(&vBuffer, r1 * cos_ta, r1 * sin_ta, -width * 0.5f, normal);
        ix3 = newVertex(&vBuffer, r2 * cos_ta_1da, r2 * sin_ta_1da, -width * 0.5f, normal);
        newFace(&iBuffer, ix0, ix1, ix2);
        newFace(&iBuffer, ix1, ix3, ix2);

        // draw outward faces of teeth
        normal = glm::vec3(v1, -u1, 0.0);
        ix0 = newVertex(&vBuffer, r1 * cos_ta, r1 * sin_ta, width * 0.5f, normal);
        ix1 = newVertex(&vBuffer, r1 * cos_ta, r1 * sin_ta, -width * 0.5f, normal);
        ix2 = newVertex(&vBuffer, r2 * cos_ta_1da, r2 * sin_ta_1da, width * 0.5f, normal);
        ix3 = newVertex(&vBuffer, r2 * cos_ta_1da, r2 * sin_ta_1da, -width * 0.5f, normal);
        newFace(&iBuffer, ix0, ix1, ix2);
        newFace(&iBuffer, ix1, ix3, ix2);

        normal = glm::vec3(cos_ta, sin_ta, 0.0);
        ix0 = newVertex(&vBuffer, r2 * cos_ta_1da, r2 * sin_ta_1da, width * 0.5f, normal);
        ix1 = newVertex(&vBuffer, r2 * cos_ta_1da, r2 * sin_ta_1da, -width * 0.5f, normal);
        ix2 = newVertex(&vBuffer, r2 * cos_ta_2da, r2 * sin_ta_2da, width * 0.5f, normal);
        ix3 = newVertex(&vBuffer, r2 * cos_ta_2da, r2 * sin_ta_2da, -width * 0.5f, normal);
        newFace(&iBuffer, ix0, ix1, ix2);
        newFace(&iBuffer, ix1, ix3, ix2);

        normal = glm::vec3(v2, -u2, 0.0);
        ix0 = newVertex(&vBuffer, r2 * cos_ta_2da, r2 * sin_ta_2da, width * 0.5f, normal);
        ix1 = newVertex(&vBuffer, r2 * cos_ta_2da, r2 * sin_ta_2da, -width * 0.5f, normal);
        ix2 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, width * 0.5f, normal);
        ix3 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, -width * 0.5f, normal);
        newFace(&iBuffer, ix0, ix1, ix2);
        newFace(&iBuffer, ix1, ix3, ix2);

        normal = glm::vec3(cos_ta, sin_ta, 0.0);
        ix0 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, width * 0.5f, normal);
        ix1 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, -width * 0.5f, normal);
        ix2 = newVertex(&vBuffer, r1 * cos_ta_4da, r1 * sin_ta_4da, width * 0.5f, normal);
        ix3 = newVertex(&vBuffer, r1 * cos_ta_4da, r1 * sin_ta_4da, -width * 0.5f, normal);
        newFace(&iBuffer, ix0, ix1, ix2);
        newFace(&iBuffer, ix1, ix3, ix2);

        // draw inside radius cylinder
        ix0 = newVertex(&vBuffer, r0 * cos_ta, r0 * sin_ta, -width * 0.5f, glm::vec3(-cos_ta, -sin_ta, 0.0));
        ix1 = newVertex(&vBuffer, r0 * cos_ta, r0 * sin_ta, width * 0.5f, glm::vec3(-cos_ta, -sin_ta, 0.0));
        ix2 = newVertex(&vBuffer, r0 * cos_ta_4da, r0 * sin_ta_4da, -width * 0.5f, glm::vec3(-cos_ta_4da, -sin_ta_4da, 0.0));
        ix3 = newVertex(&vBuffer, r0 * cos_ta_4da, r0 * sin_ta_4da, width * 0.5f, glm::vec3(-cos_ta_4da, -sin_ta_4da, 0.0));
        newFace(&iBuffer, ix0, ix1, ix2);
        newFace(&iBuffer, ix1, ix3, ix2);
    }

    // Generate vertex & index buffers
    meshInfo.vertices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vBuffer);
    meshInfo.indices = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndexBuffer, iBuffer);
    meshInfo.indexCount = (uint32_t)iBuffer.size();

    // Vertex shader uniform buffer block
    uniformData = context.createUniformBuffer(ubo);
}

void VulkanGear::draw(vk::CommandBuffer cmdbuffer, vk::PipelineLayout pipelineLayout) {
    vk::DeviceSize offsets = 0;
    cmdbuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
    cmdbuffer.bindVertexBuffers(0, meshInfo.vertices.buffer, offsets);
    cmdbuffer.bindIndexBuffer(meshInfo.indices.buffer, 0, vk::IndexType::eUint32);
    cmdbuffer.drawIndexed(meshInfo.indexCount, 1, 0, 0, 1);
}

void VulkanGear::updateUniformBuffer(const glm::mat4& perspective, const glm::mat4& view, float timer) {
    ubo.projection = perspective;
    ubo.view = view;  // glm::lookAt(glm::vec3(0, 0, -zoom), glm::vec3(-1.0, -1.5, 0), glm::vec3(0, 1, 0)) * glm::mat4_cast(orientation);
    ubo.model = glm::translate(glm::mat4(), pos) * glm::mat4_cast(glm::angleAxis(glm::radians((rotSpeed * timer) + rotOffset), glm::vec3(0, 0, 1)));
    ubo.normal = glm::inverseTranspose(ubo.view * ubo.model);

    //ubo.lightPos = lightPos;
    ubo.lightPos = glm::vec3(0.0f, 0.0f, 2.5f);
    ubo.lightPos.x = sin(glm::radians(timer)) * 8.0f;
    ubo.lightPos.z = cos(glm::radians(timer)) * 8.0f;

    uniformData.copy(ubo);
}

void VulkanGear::setupDescriptorSet(vk::DescriptorPool pool, vk::DescriptorSetLayout descriptorSetLayout) {
    descriptorSet = device.allocateDescriptorSets({ pool, 1, &descriptorSetLayout })[0];

    // Binding 0 : Vertex shader uniform buffer
    device.updateDescriptorSets({ { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.descriptor } }, nullptr);
}
