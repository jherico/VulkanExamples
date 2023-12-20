/*
* Vulkan Example - Instanced mesh rendering, uses a separate vertex buffer for instanced data
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <vulkanExampleBase.h>
#include <shapes.h>
#include <easings.hpp>
#include <glm/gtc/quaternion.hpp>

#define SHAPES_COUNT 5
#define INSTANCES_PER_SHAPE 4000
#define INSTANCE_COUNT (INSTANCES_PER_SHAPE * SHAPES_COUNT)
using namespace vk;

class VulkanExample : public vkx::ExampleBase {
public:
    vks::Buffer meshes;

    // Per-instance data block
    struct InstanceData {
        glm::vec3 pos;
        glm::vec3 rot;
        float scale;
    };

    struct ShapeVertexData {
        size_t baseVertex;
        size_t vertices;
    };

    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec3 color;
    };

    // Contains the instanced data
    vks::Buffer instanceBuffer;

    // Contains the instanced data
    vks::Buffer indirectBuffer;

    struct UboVS {
        glm::mat4 projection;
        glm::mat4 view;
        float time = 0.0f;
    } uboVS;

    struct {
        vks::Buffer vsScene;
    } uniformData;

    struct {
        vk::Pipeline solid;
    } pipelines;

    std::vector<ShapeVertexData> shapes;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    VulkanExample() {
        rotationSpeed = 0.25f;
        title = "Vulkan Example - Instanced mesh rendering";
        srand((unsigned int)time(NULL));
    }

    ~VulkanExample() {
        device.destroyPipeline(pipelines.solid);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyDescriptorSetLayout(descriptorSetLayout);
        instanceBuffer.destroy();
        indirectBuffer.destroy();
        uniformData.vsScene.destroy();
        meshes.destroy();
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& cmdBuffer) override {
        cmdBuffer.setViewport(0, vks::util::viewport(size));
        cmdBuffer.setScissor(0, vks::util::rect2D(size));
        cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
        cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.solid);
        // Binding point 0 : Mesh vertex buffer
        cmdBuffer.bindVertexBuffers(0, meshes.buffer, { 0 });
        // Binding point 1 : Instance data buffer
        cmdBuffer.bindVertexBuffers(1, instanceBuffer.buffer, { 0 });
        // Equivlant non-indirect commands:
        //for (size_t j = 0; j < SHAPES_COUNT; ++j) {
        //    auto shape = shapes[j];
        //    cmdBuffer.draw(shape.vertices, INSTANCES_PER_SHAPE, shape.baseVertex, j * INSTANCES_PER_SHAPE);
        //}
        cmdBuffer.drawIndirect(indirectBuffer.buffer, 0, SHAPES_COUNT, sizeof(vk::DrawIndirectCommand));
    }

    template <size_t N>
    void appendShape(const geometry::Solid<N>& solid, std::vector<Vertex>& vertices) {
        using namespace geometry;
        using namespace glm;
        using namespace std;
        ShapeVertexData shape;
        shape.baseVertex = vertices.size();

        auto faceCount = solid.faces.size();
        // FIXME triangulate the faces
        auto faceTriangles = triangulatedFaceTriangleCount<N>();
        vertices.reserve(vertices.size() + 3 * faceTriangles);

        vec3 color = vec3(rand(), rand(), rand()) / (float)RAND_MAX;
        color = vec3(0.3f) + (0.7f * color);
        for (size_t f = 0; f < faceCount; ++f) {
            const Face<N>& face = solid.faces[f];
            vec3 normal = solid.getFaceNormal(f);
            for (size_t ft = 0; ft < faceTriangles; ++ft) {
                // Create the vertices for the face
                vertices.push_back({ vec3(solid.vertices[face[0]]), normal, color });
                vertices.push_back({ vec3(solid.vertices[face[2 + ft]]), normal, color });
                vertices.push_back({ vec3(solid.vertices[face[1 + ft]]), normal, color });
            }
        }
        shape.vertices = vertices.size() - shape.baseVertex;
        shapes.push_back(shape);
    }

    void loadShapes() {
        std::vector<Vertex> vertexData;
        size_t vertexCount = 0;
        appendShape<>(geometry::tetrahedron(), vertexData);
        appendShape<>(geometry::octahedron(), vertexData);
        appendShape<>(geometry::cube(), vertexData);
        appendShape<>(geometry::dodecahedron(), vertexData);
        appendShape<>(geometry::icosahedron(), vertexData);
        for (auto& vertex : vertexData) {
            vertex.position *= 0.2f;
        }
        meshes = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, vertexData);
    }

    void setupDescriptorPool() {
        // Example uses one ubo
        std::vector<vk::DescriptorPoolSize> poolSizes{
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 1),
        };

        descriptorPool = device.createDescriptorPool({ {}, 1, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        // Binding 0 : Vertex shader uniform buffer
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
        };

        descriptorSetLayout = device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        descriptorSet = device.allocateDescriptorSets({ descriptorPool, 1, &descriptorSetLayout })[0];

        // Binding 0 : Vertex shader uniform buffer
        vk::WriteDescriptorSet writeDescriptorSet;
        writeDescriptorSet.dstSet = descriptorSet;
        writeDescriptorSet.descriptorType = vk::DescriptorType::eUniformBuffer;
        writeDescriptorSet.dstBinding = 0;
        writeDescriptorSet.pBufferInfo = &uniformData.vsScene.descriptor;
        writeDescriptorSet.descriptorCount = 1;

        device.updateDescriptorSets(writeDescriptorSet, nullptr);
    }

    void preparePipelines() {
        // Instacing pipeline
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout, renderPass };
        // Load shaders
        pipelineBuilder.loadShader(getAssetPath() + "shaders/indirect/indirect.vert.spv", vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(getAssetPath() + "shaders/indirect/indirect.frag.spv", vk::ShaderStageFlagBits::eFragment);
        auto bindingDescriptions = pipelineBuilder.vertexInputState.bindingDescriptions;
        auto attributeDescriptions = pipelineBuilder.vertexInputState.attributeDescriptions;
        pipelineBuilder.vertexInputState.bindingDescriptions = {
            // Mesh vertex buffer (description) at binding point 0
            { 0, sizeof(Vertex), vk::VertexInputRate::eVertex },
            { 1, sizeof(InstanceData), vk::VertexInputRate::eInstance },
        };

        // Attribute descriptions
        // Describes memory layout and shader positions
        pipelineBuilder.vertexInputState.attributeDescriptions = {
            // Per-Vertex attributes
            { 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position) },
            { 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) },
            { 2, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal) },

            // Instanced attributes
            { 3, 1, vk::Format::eR32G32B32Sfloat, offsetof(InstanceData, pos) },
            { 4, 1, vk::Format::eR32G32B32Sfloat, offsetof(InstanceData, rot) },
            { 5, 1, vk::Format::eR32Sfloat, offsetof(InstanceData, scale) },
        };

        pipelines.solid = pipelineBuilder.create(context.pipelineCache);
    }

    void prepareIndirectData() {
        std::vector<vk::DrawIndirectCommand> indirectData;
        indirectData.resize(SHAPES_COUNT);
        for (auto i = 0; i < SHAPES_COUNT; ++i) {
            auto& drawIndirectCommand = indirectData[i];
            const auto& shapeData = shapes[i];
            drawIndirectCommand.firstInstance = i * INSTANCES_PER_SHAPE;
            drawIndirectCommand.instanceCount = INSTANCES_PER_SHAPE;
            drawIndirectCommand.firstVertex = (uint32_t)shapeData.baseVertex;
            drawIndirectCommand.vertexCount = (uint32_t)shapeData.vertices;
        }
        indirectBuffer = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndirectBuffer, indirectData);
    }

    void prepareInstanceData() {
        std::vector<InstanceData> instanceData;
        instanceData.resize(INSTANCE_COUNT);

        std::mt19937 rndGenerator((unsigned int)time(nullptr));
        std::uniform_real_distribution<float> uniformDist(0.0, 1.0);
        std::exponential_distribution<float> expDist(1);

        for (auto i = 0; i < INSTANCE_COUNT; i++) {
            auto& instance = instanceData[i];
            instance.rot = (float)M_PI * glm::vec3(uniformDist(rndGenerator), uniformDist(rndGenerator), uniformDist(rndGenerator));
            float theta = 2.0f * (float)M_PI * uniformDist(rndGenerator);
            float phi = acos(1 - 2 * uniformDist(rndGenerator));
            instance.scale = 0.1f + expDist(rndGenerator) * 3.0f;
            instance.pos = glm::normalize(glm::vec3(sin(phi) * cos(theta), sin(theta), cos(phi)));
            instance.pos *= instance.scale * (1.0f + expDist(rndGenerator) / 2.0f) * 4.0f;
        }

        instanceBuffer = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer, instanceData);
    }

    void prepareUniformBuffers() {
        uniformData.vsScene = context.createUniformBuffer(uboVS);
        updateUniformBuffer(true);
    }

    void updateUniformBuffer(bool viewChanged) {
        if (viewChanged) {
            uboVS.projection = getProjection();
            uboVS.view = camera.matrices.view;
        }

        if (!paused) {
            uboVS.time += frameTimer * 0.05f;
        }

        memcpy(uniformData.vsScene.mapped, &uboVS, sizeof(uboVS));
    }

    void prepare() override {
        ExampleBase::prepare();
        loadShapes();
        prepareInstanceData();
        prepareIndirectData();
        prepareUniformBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        buildCommandBuffers();
        prepared = true;
    }

    const float duration = 4.0f;
    const float interval = 6.0f;
    float zoomDelta = 135;
    float zoomStart;
    float accumulator = FLT_MAX;

    void update(float delta) override {
        ExampleBase::update(delta);
        if (!paused) {
            accumulator += delta;
            if (accumulator < duration) {
                camera.position.z = easings::inOutQuint(accumulator, duration, zoomStart, zoomDelta);
                camera.setTranslation(camera.position);
                updateUniformBuffer(true);
            } else {
                updateUniformBuffer(false);
            }

            if (accumulator >= interval) {
                accumulator = 0;
                zoomStart = camera.position.z;
                if (camera.position.z < -2) {
                    zoomDelta = 135;
                } else {
                    zoomDelta = -135;
                }
            }
        }
    }

    void viewChanged() override { updateUniformBuffer(true); }
};

RUN_EXAMPLE(VulkanExample)
