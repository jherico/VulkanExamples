/*
* Heightmap terrain generator
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <glm/glm.hpp>
#include <gli/gli.hpp>

#include "vks/buffer.hpp"
#include "vks/context.hpp"
#include "vks/filesystem.hpp"

namespace vkx {
class HeightMap {
private:
    std::vector<uint16_t> heightdata;
    uint32_t dim;
    uint32_t scale;

    //vk::Device device;
    //vk::Queue copyQueue;
public:
    enum Topology
    {
        topologyTriangles,
        topologyQuads
    };

    float heightScale = 1.0f;
    float uvScale = 1.0f;

    vks::Buffer vertexBuffer;
    vks::Buffer indexBuffer;

    struct Vertex {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec2 uv;
    };

    size_t vertexBufferSize = 0;
    size_t indexBufferSize = 0;
    uint32_t indexCount = 0;

    void destroy() {
        vertexBuffer.destroy();
        indexBuffer.destroy();
    }

    float getHeight(uint32_t x, uint32_t y) {
        glm::ivec2 rpos = glm::ivec2(x, y) * glm::ivec2(scale);
        rpos.x = std::max(0, std::min(rpos.x, (int)dim - 1));
        rpos.y = std::max(0, std::min(rpos.y, (int)dim - 1));
        rpos /= glm::ivec2(scale);
        size_t offset = (rpos.x + rpos.y * dim) * scale;
        return heightdata[offset] / 65535.0f * heightScale;
    }

    void loadFromFile(const vks::Context& context, const std::string& filename, uint32_t patchsize, glm::vec3 scale, Topology topology) {
        std::shared_ptr<gli::texture2d> tex2Dptr;
        vks::file::withBinaryFileContents(filename, [&](size_t size, const void* data) {
            tex2Dptr = std::make_shared<gli::texture2d>(gli::load((const char*)data, size));
        });

        const auto& heightTex = *tex2Dptr;
        dim = static_cast<uint32_t>(heightTex.extent().x);
        heightdata.resize(dim * dim);
        memcpy(heightdata.data(), heightTex.data(), heightTex.size());
        this->scale = dim / patchsize;
        this->heightScale = scale.y;

        // Generate vertices

        std::vector<Vertex> vertices;
        vertices.resize(patchsize * patchsize * 4);

        const float wx = 2.0f;
        const float wy = 2.0f;

        for (uint32_t x = 0; x < patchsize; x++) {
            for (uint32_t y = 0; y < patchsize; y++) {
                uint32_t index = (x + y * patchsize);
                vertices[index].pos[0] = (x * wx + wx / 2.0f - (float)patchsize * wx / 2.0f) * scale.x;
                vertices[index].pos[1] = -getHeight(x, y);
                vertices[index].pos[2] = (y * wy + wy / 2.0f - (float)patchsize * wy / 2.0f) * scale.z;
                vertices[index].uv = glm::vec2((float)x / patchsize, (float)y / patchsize) * uvScale;
            }
        }

        for (uint32_t y = 0; y < patchsize; y++) {
            for (uint32_t x = 0; x < patchsize; x++) {
                float dx = getHeight(x < patchsize - 1 ? x + 1 : x, y) - getHeight(x > 0 ? x - 1 : x, y);
                if (x == 0 || x == patchsize - 1)
                    dx *= 2.0f;

                float dy = getHeight(x, y < patchsize - 1 ? y + 1 : y) - getHeight(x, y > 0 ? y - 1 : y);
                if (y == 0 || y == patchsize - 1)
                    dy *= 2.0f;

                glm::vec3 A = glm::vec3(1.0f, 0.0f, dx);
                glm::vec3 B = glm::vec3(0.0f, 1.0f, dy);

                glm::vec3 normal = (glm::normalize(glm::cross(A, B)) + 1.0f) * 0.5f;

                vertices[x + y * patchsize].normal = glm::vec3(normal.x, normal.z, normal.y);
            }
        }

        // Generate indices

        const uint32_t w = (patchsize - 1);
        std::vector<uint32_t> indices;

        switch (topology) {
                // Indices for triangles
            case topologyTriangles: {
                indices.resize(w * w * 6);
                for (uint32_t x = 0; x < w; x++) {
                    for (uint32_t y = 0; y < w; y++) {
                        uint32_t index = (x + y * w) * 6;
                        indices[index] = (x + y * patchsize);
                        indices[index + 1] = indices[index] + patchsize;
                        indices[index + 2] = indices[index + 1] + 1;

                        indices[index + 3] = indices[index + 1] + 1;
                        indices[index + 4] = indices[index] + 1;
                        indices[index + 5] = indices[index];
                    }
                }
                indexCount = (patchsize - 1) * (patchsize - 1) * 6;
                indexBufferSize = (w * w * 6) * sizeof(uint32_t);
                break;
            }
            // Indices for quad patches (tessellation)
            case topologyQuads: {
                indices.resize(w * w * 4);
                for (uint32_t x = 0; x < w; x++) {
                    for (uint32_t y = 0; y < w; y++) {
                        uint32_t index = (x + y * w) * 4;
                        indices[index] = (x + y * patchsize);
                        indices[index + 1] = indices[index] + patchsize;
                        indices[index + 2] = indices[index + 1] + 1;
                        indices[index + 3] = indices[index] + 1;
                    }
                }
                indexCount = (patchsize - 1) * (patchsize - 1) * 4;
                indexBufferSize = (w * w * 4) * sizeof(uint32_t);
                break;
            }
        }

        assert(indexBufferSize > 0);

        vertexBufferSize = (patchsize * patchsize * 4) * sizeof(Vertex);
        vertexBuffer = context.stageToDeviceBuffer<Vertex>(vk::BufferUsageFlagBits::eVertexBuffer, vertices);
        indexBuffer = context.stageToDeviceBuffer<uint32_t>(vk::BufferUsageFlagBits::eIndexBuffer, indices);
    }
};
}  // namespace vkx
