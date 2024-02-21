/*
 * Vulkan Example - Instanced mesh rendering, uses a separate vertex buffer for instanced data
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include <examples/example.hpp>

#include <common/easings.hpp>
#include <common/random.hpp>
#include <glm/gtc/quaternion.hpp>
#include <rendering/shapes.hpp>

#include <shaders/indirect/indirect.frag.inl>
#include <shaders/indirect/indirect.vert.inl>

#define SHAPES_COUNT 5
#define INSTANCES_PER_SHAPE 4000
#define INSTANCE_COUNT (INSTANCES_PER_SHAPE * SHAPES_COUNT)
using namespace vk;

class VulkanExample : public vkx::ExampleBase {
    using Parent = vkx::ExampleBase;

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
        device.destroy(pipelines.solid);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        instanceBuffer.destroy();
        indirectBuffer.destroy();
        uniformData.vsScene.destroy();
        meshes.destroy();
    }

    void getEnabledFeatures() override {
        Parent::getEnabledFeatures();
        context.enabledFeatures.core10.multiDrawIndirect = context.deviceInfo.features.core10.multiDrawIndirect;
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
        // for (size_t j = 0; j < SHAPES_COUNT; ++j) {
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
        meshes = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eVertexBuffer, vertexData);
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
        vks::pipelines::GraphicsPipelineBuilder pipelineBuilder{ device, pipelineLayout };
        pipelineBuilder.depthStencilState = true;
        pipelineBuilder.dynamicRendering(defaultColorFormat, defaultDepthStencilFormat);
        // Load shaders
        pipelineBuilder.loadShader(vkx::shaders::indirect::indirect::vert, vk::ShaderStageFlagBits::eVertex);
        pipelineBuilder.loadShader(vkx::shaders::indirect::indirect::frag, vk::ShaderStageFlagBits::eFragment);
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
        indirectBuffer = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eIndirectBuffer, indirectData);
    }

    void prepareInstanceData() {
        std::vector<InstanceData> instanceData;
        instanceData.resize(INSTANCE_COUNT);
        vkx::Random random;

        for (auto i = 0; i < INSTANCE_COUNT; i++) {
            auto& instance = instanceData[i];
            instance.rot = random.v3((float)M_PI);
            instance.scale = 0.1f + random.exp() * 3.0f;
            auto scale = instance.scale * (1.0f + random.exp() / 2.0f) * 4.0f;
            instance.pos = random.sphere(glm::vec3{ scale });
        }

        instanceBuffer = loader.stageToDeviceBuffer(graphicsQueue, vk::BufferUsageFlagBits::eVertexBuffer, instanceData);
    }

    void prepareUniformBuffers() {
        uniformData.vsScene = loader.createUniformBuffer(uboVS);
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
        uniformData.vsScene.copy(uboVS);
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
