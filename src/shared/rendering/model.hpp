/*
* Vulkan Model loader using ASSIMP
*
* Copyright(C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license(MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <vulkan/vulkan.hpp>
#include <vks/pipelines.hpp>
#include <vks/buffer.hpp>

struct aiScene;
namespace Assimp {
class Importer;
};

namespace vks { namespace model {

/** @brief Vertex layout components */
enum Component
{
    VERTEX_COMPONENT_POSITION = 0x0,
    VERTEX_COMPONENT_NORMAL = 0x1,
    VERTEX_COMPONENT_COLOR = 0x2,
    VERTEX_COMPONENT_UV = 0x3,
    VERTEX_COMPONENT_TANGENT = 0x4,
    VERTEX_COMPONENT_BITANGENT = 0x5,
    VERTEX_COMPONENT_DUMMY_FLOAT = 0x6,
    VERTEX_COMPONENT_DUMMY_INT = 0x7,
    VERTEX_COMPONENT_DUMMY_VEC4 = 0x8,
    VERTEX_COMPONENT_DUMMY_INT4 = 0x9,
    VERTEX_COMPONENT_DUMMY_UINT4 = 0xA,
};

/** @brief Stores vertex layout components for model loading and Vulkan vertex input and atribute bindings  */
struct VertexLayout {
public:
    void appendVertexLayout(vks::pipelines::PipelineVertexInputStateCreateInfo& vertexInputState,
                            uint32_t binding = 0,
                            vk::VertexInputRate rate = vk::VertexInputRate::eVertex) const {
        vertexInputState.bindingDescriptions.emplace_back(binding, stride(), rate);
        auto componentsSize = components.size();
        vertexInputState.attributeDescriptions.reserve(vertexInputState.attributeDescriptions.size() + componentsSize);
        auto attributeIndexOffset = (uint32_t)vertexInputState.attributeDescriptions.size();
        for (uint32_t i = 0; i < componentsSize; ++i) {
            const auto& component = components[i];
            const auto format = vks::model::VertexLayout::componentFormat(component);
            const auto offsetValue = offset(i);
            vertexInputState.attributeDescriptions.emplace_back(attributeIndexOffset + i, binding, format, offsetValue);
        }
    }

    /** @brief Components used to generate vertices from */
    std::vector<Component> components;
    VertexLayout() = default;
    VertexLayout(std::vector<Component>&& components, uint32_t binding = 0)
        : components(std::move(components)) {}

    uint32_t componentIndex(Component component) const {
        for (size_t i = 0; i < components.size(); ++i) {
            if (components[i] == component) {
                return (uint32_t)i;
            }
        }
        return static_cast<uint32_t>(-1);
    }

    static vk::Format componentFormat(Component component) {
        switch (component) {
            case VERTEX_COMPONENT_UV:
                return vk::Format::eR32G32Sfloat;
            case VERTEX_COMPONENT_DUMMY_FLOAT:
                return vk::Format::eR32Sfloat;
            case VERTEX_COMPONENT_DUMMY_INT:
                return vk::Format::eR32Sint;
            case VERTEX_COMPONENT_DUMMY_VEC4:
                return vk::Format::eR32G32B32A32Sfloat;
            case VERTEX_COMPONENT_DUMMY_INT4:
                return vk::Format::eR32G32B32A32Sint;
            case VERTEX_COMPONENT_DUMMY_UINT4:
                return vk::Format::eR32G32B32A32Uint;
            default:
                return vk::Format::eR32G32B32Sfloat;
        }
    }

    static uint32_t componentSize(Component component) {
        switch (component) {
            case VERTEX_COMPONENT_UV:
                return 2 * sizeof(float);
            case VERTEX_COMPONENT_DUMMY_FLOAT:
                return sizeof(float);
            case VERTEX_COMPONENT_DUMMY_INT:
                return sizeof(int);
            case VERTEX_COMPONENT_DUMMY_VEC4:
                return 4 * sizeof(float);
            case VERTEX_COMPONENT_DUMMY_INT4:
                return 4 * sizeof(int32_t);
            case VERTEX_COMPONENT_DUMMY_UINT4:
                return 4 * sizeof(uint32_t);
            default:
                // All components except the ones listed above are made up of 3 floats
                return 3 * sizeof(float);
        }
    }

    uint32_t stride() const {
        uint32_t res = 0;
        for (auto& component : components) {
            res += componentSize(component);
        }
        return res;
    }

    uint32_t offset(uint32_t index) const {
        uint32_t res = 0;
        assert(index < components.size());
        for (uint32_t i = 0; i < index; ++i) {
            res += componentSize(components[i]);
        }
        return res;
    }
};

/** @brief Used to parametrize model loading */
struct ModelCreateInfo {
    glm::vec3 center{ 0 };
    glm::vec3 scale{ 1 };
    glm::vec2 uvscale{ 1 };

    ModelCreateInfo() = default;

    ModelCreateInfo(const glm::vec3& scale, const glm::vec2& uvscale, const glm::vec3& center)
        : center(center)
        , scale(scale)
        , uvscale(uvscale) {}

    ModelCreateInfo(float scale, float uvscale, float center)
        : ModelCreateInfo(glm::vec3(scale), glm::vec2{ uvscale }, glm::vec3(center)) {}
};

struct Model {
    vk::Device device;
    Buffer vertices;
    Buffer indices;
    uint32_t indexCount = 0;
    uint32_t vertexCount = 0;
    VertexLayout layout;
    glm::vec3 scale{ 1.0f };
    glm::vec3 center{ 0.0f };
    glm::vec2 uvscale{ 1.0f };

    /** @brief Stores vertex and index base and counts for each part of a model */
    struct ModelPart {
        std::string name;
        uint32_t vertexBase;
        uint32_t vertexCount;
        uint32_t indexBase;
        uint32_t indexCount;
    };
    std::vector<ModelPart> parts;

    static const int defaultFlags;

    struct Dimension {
        glm::vec3 min = glm::vec3(FLT_MAX);
        glm::vec3 max = glm::vec3(-FLT_MAX);
        glm::vec3 size;
    } dim;

    Model() = default;
    Model(const Model&) = delete;
    Model(Model&&) = default;
    /** @brief Release all Vulkan resources of this model */
    void destroy() {
        vertices.destroy();
        indices.destroy();
    }

    /**
    * Loads a 3D model from a file into Vulkan buffers
    *
    * @param device Pointer to the Vulkan device used to generated the vertex and index buffers on
    * @param filename File to load (must be a model format supported by ASSIMP)
    * @param layout Vertex layout components (position, normals, tangents, etc.)
    * @param createInfo MeshCreateInfo structure for load time settings like scale, center, etc.
    * @param (Optional) flags ASSIMP model loading flags
    */
    void loadFromFile(const std::string& filename, const VertexLayout& layout, const ModelCreateInfo& createInfo, int flags = defaultFlags);

    /**
    * Loads a 3D model from a file into Vulkan buffers
    *
    * @param device Pointer to the Vulkan device used to generated the vertex and index buffers on
    * @param filename File to load (must be a model format supported by ASSIMP)
    * @param layout Vertex layout components (position, normals, tangents, etc.)
    * @param scale Load time scene scale
    * @param (Optional) flags ASSIMP model loading flags
    */
    void loadFromFile(const std::string& filename, const VertexLayout& layout, float scale = 1.0f, const int flags = defaultFlags) {
        loadFromFile(filename, layout, ModelCreateInfo{ scale, 1.0f, 0.0f }, flags);
    }

    virtual void onLoad(Assimp::Importer& importer, const aiScene* pScene) {}

    virtual void appendVertex(std::vector<uint8_t>& outputBuffer, const aiScene* pScene, uint32_t meshIndex, uint32_t vertexIndex);

    template <typename T>
    void appendOutput(std::vector<uint8_t>& outputBuffer, const T& t) {
        auto offset = outputBuffer.size();
        auto copySize = sizeof(T);
        outputBuffer.resize(offset + copySize);
        memcpy(outputBuffer.data() + offset, &t, copySize);
    }

    template <typename T>
    void appendOutput(std::vector<uint8_t>& outputBuffer, const std::vector<T>& v) {
        auto offset = outputBuffer.size();
        auto copySize = v.size() * sizeof(T);
        outputBuffer.resize(offset + copySize);
        memcpy(outputBuffer.data() + offset, v.data(), copySize);
    }
};

}}  // namespace vks::model
