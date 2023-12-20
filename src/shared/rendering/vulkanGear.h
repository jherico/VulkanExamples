/*
* Vulkan Example - Animated gears using multiple uniform buffers
*
* See readme.md for details
*
* Copyright (C) 2015 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <glm/glm.hpp>
#include "vks/context.hpp"

struct Vertex {
    float pos[3];
    float normal[3];
    float color[3];

    Vertex(const glm::vec3& p, const glm::vec3& n, const glm::vec3& c) {
        pos[0] = p.x;
        pos[1] = p.y;
        pos[2] = p.z;
        color[0] = c.x;
        color[1] = c.y;
        color[2] = c.z;
        normal[0] = n.x;
        normal[1] = n.y;
        normal[2] = n.z;
    }
};

class VulkanGear {
private:
    struct UBO {
        glm::mat4 projection;
        glm::mat4 model;
        glm::mat4 normal;
        glm::mat4 view;
        glm::vec3 lightPos;
    };

    vk::Device device;
    glm::vec3 color;
    glm::vec3 pos;
    float rotSpeed;
    float rotOffset;

    struct MeshInfo {
        vks::Buffer vertices;
        vks::Buffer indices;
        uint32_t indexCount;

        void destroy() {
            vertices.destroy();
            indices.destroy();
        }
    } meshInfo;

    UBO ubo;
    vks::Buffer uniformData;

    int32_t newVertex(std::vector<Vertex>* vBuffer, float x, float y, float z, const glm::vec3& normal);
    void newFace(std::vector<uint32_t>* iBuffer, int a, int b, int c);

public:
    vk::DescriptorSet descriptorSet;

    void draw(vk::CommandBuffer cmdbuffer, vk::PipelineLayout pipelineLayout);
    void updateUniformBuffer(const glm::mat4& perspective, const glm::mat4& view, float timer);

    void setupDescriptorSet(vk::DescriptorPool pool, vk::DescriptorSetLayout descriptorSetLayout);

    ~VulkanGear();

    void generate(const vks::Context& context,
                  float inner_radius,
                  float outer_radius,
                  float width,
                  int teeth,
                  float tooth_depth,
                  glm::vec3 color,
                  glm::vec3 pos,
                  float rotSpeed,
                  float rotOffset);
};
