/*
* Vulkan Example - Compute shader culling and LOD using indirect rendering
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*
*/

#include <vulkanExampleBase.h>
#include <vks/frustum.hpp>

// Total number of objects (^3) in the scene
#if defined(__ANDROID__)
#define OBJECT_COUNT 32
#else
#define OBJECT_COUNT 64
#endif

#define MAX_LOD_LEVEL 5

// Resources for the compute part of the example
struct Compute : public vkx::Compute {
    using Parent = vkx::Compute;
    Compute(const vks::Context& context)
        : vkx::Compute(context) {}

    uint32_t objectCount = 0;

    struct {
        vks::model::Model lodObject;
    } models;

    struct {
        vks::Buffer scene;
    } uniformData;

    vks::Buffer lodLevelsBuffers;  // Contains index start and counts for the different lod levels
    // Contains the instanced data
    vks::Buffer instanceBuffer;
    // Contains the indirect drawing commands
    vks::Buffer indirectCommandsBuffer;
    vks::Buffer indirectDrawCountBuffer;

    //vk::Fence fence;
    vk::CommandBuffer commandBuffer;
    vk::DescriptorPool descriptorPool;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::DescriptorSet descriptorSet;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;

    void prepare() override {
        Parent::prepare();
        // Create a command buffer for compute operations
        commandBuffer = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{ commandPool, vk::CommandBufferLevel::ePrimary, 1 })[0];
        prepareDescriptors();
        preparePipeline();
        buildCommandBuffer();
    }

    void destroy() override {
        models.lodObject.destroy();
        instanceBuffer.destroy();
        indirectCommandsBuffer.destroy();
        uniformData.scene.destroy();
        indirectDrawCountBuffer.destroy();
        lodLevelsBuffers.destroy();

        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);
        device.destroy(pipeline);
        device.destroy(descriptorPool);
        device.freeCommandBuffers(commandPool, commandBuffer);
        Parent::destroy();
    }

    void prepareDescriptors() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 1 },
            { vk::DescriptorType::eStorageBuffer, 4 },
        };
        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });

        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0: Instance input data buffer
            { 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
            // Binding 1: Indirect draw command output buffer (input)
            { 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
            // Binding 2: Uniform buffer with global matrices (input)
            { 2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute },
            // Binding 3: Indirect draw stats (output)
            { 3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
            // Binding 4: LOD info (input)
            { 4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
        };

        descriptorSetLayout =
            context.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo{ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{ {}, 1, &descriptorSetLayout });

        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &descriptorSetLayout };

        //// Create two descriptor sets with input and output buffers switched
        descriptorSet = device.allocateDescriptorSets(allocInfo)[0];

        std::vector<vk::WriteDescriptorSet> computeWriteDescriptorSets{
            // Binding 0: Instance input data buffer
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &instanceBuffer.descriptor },
            // Binding 1: Indirect draw command output buffer
            { descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &indirectCommandsBuffer.descriptor },
            // Binding 2: Uniform buffer with global matrices
            { descriptorSet, 2, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &uniformData.scene.descriptor },
            // Binding 3: Atomic counter (written in shader)
            { descriptorSet, 3, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &indirectDrawCountBuffer.descriptor },
            // Binding 4: LOD info
            { descriptorSet, 4, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &lodLevelsBuffers.descriptor },
        };

        device.updateDescriptorSets(computeWriteDescriptorSets, nullptr);
    }

    void preparePipeline() {
        //// Push constants used to pass some parameters
        //vk::PushConstantRange pushConstantRange{ vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t) };
        //pipelineLayout = context.device.createPipelineLayout({ {}, 1, &descriptorSetLayout, 1, &pushConstantRange });

        //// Create pipeline
        //vk::ComputePipelineCreateInfo computePipelineCreateInfo;
        //computePipelineCreateInfo.layout = pipelineLayout;
        //computePipelineCreateInfo.stage =
        //    vks::shaders::loadShader(context.device, vkx::getAssetPath() + "shaders/computecloth/cloth.comp.spv", vk::ShaderStageFlagBits::eCompute);
        //pipeline = device.createComputePipeline(context.pipelineCache, computePipelineCreateInfo);
        //device.destroyShaderModule(computePipelineCreateInfo.stage.module);
        vk::ComputePipelineCreateInfo computePipelineCreateInfo;
        computePipelineCreateInfo.layout = pipelineLayout;
        computePipelineCreateInfo.stage =
            vks::shaders::loadShader(context.device, vkx::getAssetPath() + "shaders/computecullandlod/cull.comp.spv", vk::ShaderStageFlagBits::eCompute);

        // Use specialization constants to pass max. level of detail (determined by no. of meshes)
        vk::SpecializationMapEntry specializationEntry;
        specializationEntry.constantID = 0;
        specializationEntry.offset = 0;
        specializationEntry.size = sizeof(uint32_t);

        uint32_t specializationData = static_cast<uint32_t>(models.lodObject.parts.size()) - 1;

        vk::SpecializationInfo specializationInfo;
        specializationInfo.mapEntryCount = 1;
        specializationInfo.pMapEntries = &specializationEntry;
        specializationInfo.dataSize = sizeof(specializationData);
        specializationInfo.pData = &specializationData;

        computePipelineCreateInfo.stage.pSpecializationInfo = &specializationInfo;
        pipeline = device.createComputePipeline(context.pipelineCache, computePipelineCreateInfo).value;
        device.destroyShaderModule(computePipelineCreateInfo.stage.module);
    }

    void buildCommandBuffer() {
        commandBuffer.begin({ vk::CommandBufferUsageFlagBits::eSimultaneousUse });

        // Add memory barrier to ensure that the indirect commands have been consumed before the compute shader updates them
        vk::BufferMemoryBarrier bufferBarrier;
        bufferBarrier.buffer = indirectCommandsBuffer.buffer;
        bufferBarrier.size = indirectCommandsBuffer.descriptor.range;
        bufferBarrier.srcAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
        bufferBarrier.dstAccessMask = vk::AccessFlagBits::eShaderWrite;
        bufferBarrier.srcQueueFamilyIndex = context.queueIndices.graphics;
        bufferBarrier.dstQueueFamilyIndex = context.queueIndices.compute;
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eDrawIndirect, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, bufferBarrier, nullptr);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout, 0, descriptorSet, nullptr);

        // Dispatch the compute job
        // The compute shader will do the frustum culling and adjust the indirect draw calls depending on object visibility.
        // It also determines the lod to use depending on distance to the viewer.
        commandBuffer.dispatch(objectCount / 16, 1, 1);

        // Add memory barrier to ensure that the compute shader has finished writing the indirect command buffer before it's consumed
        std::swap(bufferBarrier.srcAccessMask, bufferBarrier.dstAccessMask);
        std::swap(bufferBarrier.srcQueueFamilyIndex, bufferBarrier.dstQueueFamilyIndex);
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eDrawIndirect, {}, nullptr, bufferBarrier, nullptr);

        // todo: barrier for indirect stats buffer?
        commandBuffer.end();
    }

    void submit() { Parent::submit(commandBuffer); }
};

class VulkanExample : public vkx::ExampleBase {
public:
    bool fixedFrustum = false;

    // Vertex layout for the models
    vks::model::VertexLayout vertexLayout = vks::model::VertexLayout({
        vks::model::VERTEX_COMPONENT_POSITION,
        vks::model::VERTEX_COMPONENT_NORMAL,
        vks::model::VERTEX_COMPONENT_COLOR,
    });

    // Per-instance data block
    struct InstanceData {
        glm::vec3 pos;
        float scale;
    };

    // Indirect draw statistics (updated via compute)
    struct {
        uint32_t drawCount;                    // Total number of indirect draw counts to be issued
        uint32_t lodCount[MAX_LOD_LEVEL + 1];  // Statistics for number of draws per LOD level (written by compute shader)
    } indirectStats;

    // Store the indirect draw commands containing index offsets and instance count per object
    std::vector<vk::DrawIndexedIndirectCommand> indirectCommands;

    struct {
        glm::mat4 projection;
        glm::mat4 modelview;
        glm::vec4 cameraPos;
        glm::vec4 frustumPlanes[6];
    } uboScene;

    struct {
        vk::Pipeline plants;
    } pipelines;

    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSet descriptorSet;
    vk::DescriptorSetLayout descriptorSetLayout;

    Compute compute{ context };

    //// Resources for the compute part of the example
    //struct {
    //    vk::Queue queue;                              // Separate queue for compute commands (queue family may differ from the one used for graphics)
    //    vk::CommandPool commandPool;                  // Use a separate command pool (queue family may differ from the one used for graphics)
    //    vk::CommandBuffer commandBuffer;              // Command buffer storing the dispatch commands and barriers
    //    vk::Fence fence;                              // Synchronization fence to avoid rewriting compute CB if still in use
    //    vk::Semaphore semaphore;                      // Used as a wait semaphore for graphics submission
    //    vk::DescriptorSetLayout descriptorSetLayout;  // Compute shader binding layout
    //    vk::DescriptorSet descriptorSet;              // Compute shader bindings
    //    vk::PipelineLayout pipelineLayout;            // Layout of the compute pipeline
    //    vk::Pipeline pipeline;                        // Compute pipeline for updating particle positions
    //} compute;

    // View frustum for culling invisible objects
    vks::Frustum frustum;

    VulkanExample() {
        title = "Vulkan Example - Compute cull and lod";
        camera.type = Camera::CameraType::firstperson;
        camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 512.0f);
        camera.setTranslation(glm::vec3(0.5f, 0.0f, 0.0f));
        camera.movementSpeed = 5.0f;
        settings.overlay = true;
        memset(&indirectStats, 0, sizeof(indirectStats));
    }

    ~VulkanExample() {
        device.destroy(pipelines.plants);
        device.destroy(pipelineLayout);
        device.destroy(descriptorSetLayout);

        compute.destroy();
    }

    void getEnabledFeatures() override {
        // Enable multi draw indirect if supported
        if (context.deviceFeatures.multiDrawIndirect) {
            context.enabledFeatures.multiDrawIndirect = VK_TRUE;
        }
    }

    void updateDrawCommandBuffer(const vk::CommandBuffer& drawCommandBuffer) override {
        drawCommandBuffer.setViewport(0, viewport());
        drawCommandBuffer.setScissor(0, scissor());
        drawCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);

        // Mesh containing the LODs
        drawCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.plants);
        drawCommandBuffer.bindVertexBuffers(0, compute.models.lodObject.vertices.buffer, { 0 });
        drawCommandBuffer.bindVertexBuffers(1, compute.instanceBuffer.buffer, { 0 });
        drawCommandBuffer.bindIndexBuffer(compute.models.lodObject.indices.buffer, 0, vk::IndexType::eUint32);

        if (context.deviceFeatures.multiDrawIndirect) {
            drawCommandBuffer.drawIndexedIndirect(compute.indirectCommandsBuffer.buffer, 0, indirectStats.drawCount, sizeof(VkDrawIndexedIndirectCommand));
        } else {
            // If multi draw is not available, we must issue separate draw commands
            for (auto j = 0; j < indirectCommands.size(); j++) {
                drawCommandBuffer.drawIndexedIndirect(compute.indirectCommandsBuffer.buffer, j * sizeof(VkDrawIndexedIndirectCommand), 1,
                                                      sizeof(VkDrawIndexedIndirectCommand));
            }
        }
    }
#if 0
    void buildCommandBuffers() {
        VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

        VkClearValue clearValues[2];
        clearValues[0].color = { { 0.18f, 0.27f, 0.5f, 0.0f } };
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.renderArea.extent.width = width;
        renderPassBeginInfo.renderArea.extent.height = height;
        renderPassBeginInfo.clearValueCount = 2;
        renderPassBeginInfo.pClearValues = clearValues;

        for (int32_t i = 0; i < drawCmdBuffers.size(); ++i) {
            // Set target frame buffer
            renderPassBeginInfo.framebuffer = frameBuffers[i];

            VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

            vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);



            vkCmdEndRenderPass(drawCmdBuffers[i]);

            VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
        }
    }
#endif

    void loadAssets() override { compute.models.lodObject.loadFromFile(context, getAssetPath() + "models/suzanne_lods.dae", vertexLayout, 0.1f); }

    void setupDescriptorPool() {
        std::vector<vk::DescriptorPoolSize> poolSizes = {
            { vk::DescriptorType::eUniformBuffer, 1 },
        };

        descriptorPool = device.createDescriptorPool({ {}, 2, (uint32_t)poolSizes.size(), poolSizes.data() });
    }

    void setupDescriptorSetLayout() {
        std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{
            // Binding 0: Vertex shader uniform buffer
            { 0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex },
        };

        descriptorSetLayout = context.device.createDescriptorSetLayout({ {}, (uint32_t)setLayoutBindings.size(), setLayoutBindings.data() });
        pipelineLayout = device.createPipelineLayout({ {}, 1, &descriptorSetLayout });
    }

    void setupDescriptorSet() {
        vk::DescriptorSetAllocateInfo allocInfo{ descriptorPool, 1, &descriptorSetLayout };
        descriptorSet = device.allocateDescriptorSets(allocInfo)[0];
        std::vector<vk::WriteDescriptorSet> writeDescriptorSets{
            // Binding 0: Vertex shader uniform buffer
            { descriptorSet, 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &compute.uniformData.scene.descriptor },
        };
        device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }

    void preparePipelines() {
        vks::pipelines::GraphicsPipelineBuilder builder{ device, pipelineLayout, renderPass };
        builder.rasterizationState.frontFace = vk::FrontFace::eClockwise;

        // Indirect (and instanced) pipeline for the plants
        builder.loadShader(getAssetPath() + "shaders/computecullandlod/indirectdraw.vert.spv", vk::ShaderStageFlagBits::eVertex);
        builder.loadShader(getAssetPath() + "shaders/computecullandlod/indirectdraw.frag.spv", vk::ShaderStageFlagBits::eFragment);
        // Per-vertex data
        builder.vertexInputState.appendVertexLayout(vertexLayout);
        builder.vertexInputState.bindingDescriptions.push_back({ 1, sizeof(InstanceData), vk::VertexInputRate::eInstance });
        // Location 4: Position
        builder.vertexInputState.attributeDescriptions.push_back(
            vk::VertexInputAttributeDescription{ 4, 1, vk::Format::eR32G32B32Sfloat, offsetof(InstanceData, pos) });
        // Location 5: Scale
        builder.vertexInputState.attributeDescriptions.push_back(
            vk::VertexInputAttributeDescription{ 5, 1, vk::Format::eR32Sfloat, offsetof(InstanceData, scale) });
        pipelines.plants = builder.create(context.pipelineCache);
    }

    void prepareBuffers() {
        compute.objectCount = OBJECT_COUNT * OBJECT_COUNT * OBJECT_COUNT;
        indirectCommands.resize(compute.objectCount);

        // Indirect draw commands
        for (uint32_t x = 0; x < OBJECT_COUNT; x++) {
            for (uint32_t y = 0; y < OBJECT_COUNT; y++) {
                for (uint32_t z = 0; z < OBJECT_COUNT; z++) {
                    uint32_t index = x + y * OBJECT_COUNT + z * OBJECT_COUNT * OBJECT_COUNT;
                    indirectCommands[index].instanceCount = 1;
                    indirectCommands[index].firstInstance = index;
                    // firstIndex and indexCount are written by the compute shader
                }
            }
        }

        indirectStats.drawCount = static_cast<uint32_t>(indirectCommands.size());
        compute.indirectCommandsBuffer =
            context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eStorageBuffer, indirectCommands);
        compute.indirectDrawCountBuffer =
            context.createBuffer(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
                                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, sizeof(indirectStats));
        // Map for host access
        compute.indirectDrawCountBuffer.map();

        // Instance data
        std::vector<InstanceData> instanceData(compute.objectCount);
        for (uint32_t x = 0; x < OBJECT_COUNT; x++) {
            for (uint32_t y = 0; y < OBJECT_COUNT; y++) {
                for (uint32_t z = 0; z < OBJECT_COUNT; z++) {
                    uint32_t index = x + y * OBJECT_COUNT + z * OBJECT_COUNT * OBJECT_COUNT;
                    instanceData[index].pos = glm::vec3((float)x, (float)y, (float)z) - glm::vec3((float)OBJECT_COUNT / 2.0f);
                    instanceData[index].scale = 2.0f;
                }
            }
        }
        compute.instanceBuffer = context.stageToDeviceBuffer(vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer, instanceData);

        // Shader storage buffer containing index offsets and counts for the LODs
        struct LOD {
            uint32_t firstIndex;
            uint32_t indexCount;
            float distance;
            float _pad0;
        };
        std::vector<LOD> LODLevels;
        uint32_t n = 0;
        for (auto modelPart : compute.models.lodObject.parts) {
            LOD lod;
            lod.firstIndex = modelPart.indexBase;   // First index for this LOD
            lod.indexCount = modelPart.indexCount;  // Index count for this LOD
            lod.distance = 5.0f + n * 5.0f;         // Starting distance (to viewer) for this LOD
            n++;
            LODLevels.push_back(lod);
        }

        compute.lodLevelsBuffers = context.stageToDeviceBuffer<LOD>(vk::BufferUsageFlagBits::eStorageBuffer, LODLevels);

        // Scene uniform buffer
        compute.uniformData.scene = context.createUniformBuffer(uboScene);
        updateUniformBuffer(true);
    }

    void updateUniformBuffer(bool viewChanged) {
        if (viewChanged) {
            uboScene.projection = camera.matrices.perspective;
            uboScene.modelview = camera.matrices.view;
            if (!fixedFrustum) {
                uboScene.cameraPos = glm::vec4(camera.position, 1.0f) * -1.0f;
                frustum.update(uboScene.projection * uboScene.modelview);
                memcpy(uboScene.frustumPlanes, frustum.planes.data(), sizeof(glm::vec4) * 6);
            }
        }

        compute.uniformData.scene.copy(uboScene);
    }

    void draw() override {
        ExampleBase::prepareFrame();

        // Submit compute shader for frustum culling
        vk::SubmitInfo computeSubmitInfo;
        computeSubmitInfo.commandBufferCount = 1;
        computeSubmitInfo.pCommandBuffers = &compute.commandBuffer;
        computeSubmitInfo.signalSemaphoreCount = 1;
        computeSubmitInfo.pSignalSemaphores = &compute.semaphores.complete;
        compute.queue.submit(computeSubmitInfo, nullptr);

        // Wait on present and compute semaphores
        renderWaitSemaphores = {
            semaphores.acquireComplete,   // Wait for presentation to finished
            compute.semaphores.complete,  // Wait for compute to finish
        };
        renderWaitStages = {
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eComputeShader,
        };

        // Submit to queue
        drawCurrentCommandBuffer();
        ExampleBase::submitFrame();

        // Get draw count from compute
        memcpy(&indirectStats, compute.indirectDrawCountBuffer.mapped, sizeof(indirectStats));
    }

    void prepare() override {
        ExampleBase::prepare();
        prepareBuffers();
        setupDescriptorSetLayout();
        preparePipelines();
        setupDescriptorPool();
        setupDescriptorSet();
        compute.prepare();
        buildCommandBuffers();
        prepared = true;
    }

    void viewChanged() override { updateUniformBuffer(true); }

    void OnUpdateUIOverlay() override {
        if (ui.header("Settings")) {
            if (ui.checkBox("Freeze frustum", &fixedFrustum)) {
                updateUniformBuffer(true);
            }
        }
        if (ui.header("Statistics")) {
            ui.text("Visible objects: %d", indirectStats.drawCount);
            for (uint32_t i = 0; i < MAX_LOD_LEVEL + 1; i++) {
                ui.text("LOD %d: %d", i, indirectStats.lodCount[i]);
            }
        }
    }
};

VULKAN_EXAMPLE_MAIN()
